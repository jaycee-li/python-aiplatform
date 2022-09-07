# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import abc
from abc import abstractmethod
import inspect
import functools
import uuid
from contextlib import contextmanager
from datetime import datetime
import logging

from google.cloud import aiplatform
from google.cloud.aiplatform.metadata.metadata import _experiment_tracker
from google.cloud.aiplatform.metadata.autolog import gorilla
from google.cloud.aiplatform.metadata.autolog import (
    get_autologging_config,
    autologging_is_disabled,
)
from google.cloud.aiplatform.metadata.autolog.events import AutologgingEventLogger


_logger = logging.getLogger(__name__)

_AUTOLOGGING_PATCHES = {}
_AUTOLOGGING_TEST_MODE_ENV_VAR = "MLFLOW_AUTOLOGGING_TESTING"


class PatchFunction:
    """
    Base class representing a function patch implementation with a callback for error handling.
    `PatchFunction` should be subclassed and used in conjunction with `safe_patch` to
    safely modify the implementation of a function. Subclasses of `PatchFunction` should
    use `_patch_implementation` to define modified ("patched") function implementations and
    `_on_exception` to define cleanup logic when `_patch_implementation` terminates due
    to an unhandled exception.
    """

    @abstractmethod
    def _patch_implementation(self, original, *args, **kwargs):
        """
        Invokes the patch function code.
        :param original: The original, underlying function over which the `PatchFunction`
                         is being applied.
        :param *args: The positional arguments passed to the original function.
        :param **kwargs: The keyword arguments passed to the original function.
        """
        pass

    @abstractmethod
    def _on_exception(self, exception):
        """
        Called when an unhandled standard Python exception (i.e. an exception inheriting from
        `Exception`) or a `KeyboardInterrupt` prematurely terminates the execution of
        `_patch_implementation`.
        :param exception: The unhandled exception thrown by `_patch_implementation`.
        """
        pass

    @classmethod
    def call(cls, original, *args, **kwargs):
        return cls().__call__(original, *args, **kwargs)

    def __call__(self, original, *args, **kwargs):
        try:
            return self._patch_implementation(original, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            try:
                self._on_exception(e)
            finally:
                # Regardless of what happens during the `_on_exception` callback, reraise
                # the original implementation exception once the callback completes
                raise e


def with_managed_run(autologging_integration, patch_function):
    """
    Given a `patch_function`, returns an `augmented_patch_function` that wraps the execution of
    `patch_function` with an active MLflow run. The following properties apply:
        - An MLflow run is only created if there is no active run present when the
          patch function is executed
        - If an active run is created by the `augmented_patch_function`, it is terminated
          with the `FINISHED` state at the end of function execution
        - If an active run is created by the `augmented_patch_function`, it is terminated
          with the `FAILED` if an unhandled exception is thrown during function execution
    Note that, if nested runs or non-fluent runs are created by `patch_function`, `patch_function`
    is responsible for terminating them by the time it terminates
    (or in the event of an exception).
    :param autologging_integration: The autologging integration associated
                                    with the `patch_function`.
    :param patch_function: A `PatchFunction` class definition or a function object
                           compatible with `safe_patch`.
    :param tags: A dictionary of string tags to set on each managed run created during the
                 execution of `patch_function`.
    """

    # def create_managed_run():
    #     TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    #     managed_run = aiplatform.start_run(run=f"{autologging_integration}-autologging-run-{TIMESTAMP}")
    #     _logger.info(
    #         "Created ExperimentRun with ID '%s', which will track hyperparameters,"
    #         " performance metrics, model artifacts, and lineage information for the"
    #         " current %s workflow",
    #         managed_run.run_name,
    #         autologging_integration,
    #     )
    #     return managed_run

    if inspect.isclass(patch_function):

        class PatchWithManagedRun(patch_function):
            def __init__(self):
                super().__init__()
                self.managed_run = None

            def _patch_implementation(self, original, *args, **kwargs):
                if not _experiment_tracker.experiment_run:
                    raise ValueError(
                        "No experimentRun set. Make sure to call aiplatform.start_run('my-run') before training your model."
                    )
                    # self.managed_run = create_managed_run()

                result = super()._patch_implementation(original, *args, **kwargs)

                if self.managed_run:
                    aiplatform.end_run()

                return result

            def _on_exception(self, e):
                if self.managed_run:
                    aiplatform.end_run()
                super()._on_exception(e)

        return PatchWithManagedRun

    else:

        def patch_with_managed_run(original, *args, **kwargs):
            managed_run = None
            if not _experiment_tracker.experiment_run:
                raise ValueError(
                    "No experimentRun set. Make sure to call aiplatform.start_run('my-run') before training your model."
                )
                # managed_run = create_managed_run()

            try:
                result = patch_function(original, *args, **kwargs)
            except (Exception, KeyboardInterrupt):
                # In addition to standard Python exceptions, handle keyboard interrupts to ensure
                # that runs are terminated if a user prematurely interrupts training execution
                # (e.g. via sigint / ctrl-c)
                if managed_run:
                    aiplatform.end_run()
                raise
            else:
                if managed_run:
                    aiplatform.end_run()
                return result

        return patch_with_managed_run


def is_testing():
    """
    Indicates whether or not autologging functionality is running in test mode (as determined
    by the `MLFLOW_AUTOLOGGING_TESTING` environment variable). Test mode performs additional
    validation during autologging, including:
        - Checks for the exception safety of arguments passed to model training functions
          (i.e. all additional arguments should be "exception safe" functions or classes)
        - Disables exception handling for patched function logic, ensuring that patch code
          executes without errors during testing
    """
    return os.environ.get(_AUTOLOGGING_TEST_MODE_ENV_VAR, "false") == "true"


def safe_patch(
    autologging_integration,
    destination,
    function_name,
    patch_function,
    manage_run=False,
):
    """
    Patches the specified `function_name` on the specified `destination` class for autologging
    purposes, preceding its implementation with an error-safe copy of the specified patch
    `patch_function` with the following error handling behavior:
        - Exceptions thrown from the underlying / original function
          (`<destination>.<function_name>`) are propagated to the caller.
        - Exceptions thrown from other parts of the patched implementation (`patch_function`)
          are caught and logged as warnings.
    :param autologging_integration: The name of the autologging integration associated with the
                                    patch.
    :param destination: The Python class on which the patch is being defined.
    :param function_name: The name of the function to patch on the specified `destination` class.
    :param patch_function: The patched function code to apply. This is either a `PatchFunction`
                           class definition or a function object. If it is a function object, the
                           first argument should be reserved for an `original` method argument
                           representing the underlying / original function. Subsequent arguments
                           should be identical to those of the original function being patched.
    :param manage_run: If `True`, applies the `with_managed_run` wrapper to the specified
                       `patch_function`, which automatically creates & terminates an MLflow
                       active run during patch code execution if necessary. If `False`,
                       does not apply the `with_managed_run` wrapper to the specified
                       `patch_function`.
    """

    if manage_run:
        patch_function = with_managed_run(
            autologging_integration,
            patch_function,
        )

    patch_is_class = inspect.isclass(patch_function)
    if patch_is_class:
        assert issubclass(patch_function, PatchFunction)
    else:
        assert callable(patch_function)

    original_fn = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=False
    )
    # Retrieve raw attribute while bypassing the descriptor protocol
    raw_original_obj = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=True
    )
    if original_fn != raw_original_obj:
        raise RuntimeError(f"Unsupport patch on {str(destination)}.{function_name}")
    elif isinstance(original_fn, property):
        is_property_method = True

        # For property decorated methods (a kind of method delegation), e.g.
        # class A:
        #   @property
        #   def f1(self):
        #     ...
        #     return delegated_f1
        #
        # suppose `a1` is an instance of class `A`,
        # `A.f1.fget` will get the original `def f1(self)` method,
        # and `A.f1.fget(a1)` will be equivalent to `a1.f1()` and
        # its return value will be the `delegated_f1` function.
        # So using the `property.fget` we can construct the (delegated) "original_fn"
        def original(self, *args, **kwargs):
            # the `original_fn.fget` will get the original method decorated by `property`
            # the `original_fn.fget(self)` will get the delegated function returned by the
            # property decorated method.
            bound_delegate_method = original_fn.fget(self)
            return bound_delegate_method(*args, **kwargs)

    else:
        original = original_fn
        is_property_method = False

    def safe_patch_function(*args, **kwargs):
        """
        A safe wrapper around the specified `patch_function` implementation designed to
        handle exceptions thrown during the execution of `patch_function`. This wrapper
        distinguishes exceptions thrown from the underlying / original function
        (`<destination>.<function_name>`) from exceptions thrown from other parts of
        `patch_function`. This distinction is made by passing an augmented version of the
        underlying / original function to `patch_function` that uses nonlocal state to track
        whether or not it has been executed and whether or not it threw an exception.
        Exceptions thrown from the underlying / original function are propagated to the caller,
        while exceptions thrown from other parts of `patch_function` are caught and logged as
        warnings.
        """

        # if is_testing():
        #     preexisting_run_for_testing = _experiment_tracker.experiment_run

        # # Whether or not to exclude autologged content from user-created fluent runs
        # # (i.e. runs created manually via `mlflow.start_run()`)
        # exclusive = get_autologging_config(autologging_integration, "exclusive", False)
        # active_run = (
        #     _experiment_tracker.experiment_run
        #     and not _AutologgingSessionManager.active_session()
        # )
        # active_session_failed = (
        #     _AutologgingSessionManager.active_session() is not None
        #     and _AutologgingSessionManager.active_session().state == "failed"
        # )

        # if (
        #     active_session_failed
        #     or autologging_is_disabled(autologging_integration)
        #     or (active_run and exclusive)
        #     or mlflow.utils.autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED
        # ):
        #     # If the autologging integration associated with this patch is disabled,
        #     # or if the current autologging integration is in exclusive mode and a user-created
        #     # fluent run is active, call the original function and return. Restore the original
        #     # warning behavior during original function execution, since autologging is being
        #     # skipped
        #     with set_non_mlflow_warnings_behavior_for_current_thread(
        #         disable_warnings=False,
        #         reroute_warnings=False,
        #     ):
        #         return original(*args, **kwargs)

        # Whether or not the original / underlying function has been called during the
        # execution of patched code
        original_has_been_called = False
        # The value returned by the call to the original / underlying function during
        # the execution of patched code
        original_result = None
        # Whether or not an exception was raised from within the original / underlying function
        # during the execution of patched code
        failed_during_original = False
        # The active MLflow run (if any) associated with patch code execution
        patch_function_run_for_testing = None
        # The exception raised during executing patching function
        patch_function_exception = None

        def try_log_autologging_event(log_fn, *args):
            try:
                log_fn(*args)
            except Exception as e:
                _logger.debug(
                    "Failed to log autologging event via '%s'. Exception: %s",
                    log_fn,
                    e,
                )

        def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
            try:
                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_original_function_start,
                    session,
                    destination,
                    function_name,
                    og_args,
                    og_kwargs,
                )
                original_fn_result = original_fn(*og_args, **og_kwargs)

                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_original_function_success,
                    session,
                    destination,
                    function_name,
                    og_args,
                    og_kwargs,
                )
                return original_fn_result
            except Exception as original_fn_e:
                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_original_function_error,
                    session,
                    destination,
                    function_name,
                    og_args,
                    og_kwargs,
                    original_fn_e,
                )

                nonlocal failed_during_original
                failed_during_original = True
                raise

        with _AutologgingSessionManager.start_session(
            autologging_integration
        ) as session:
            try:

                def call_original(*og_args, **og_kwargs):
                    def _original_fn(*_og_args, **_og_kwargs):
                        nonlocal original_has_been_called
                        original_has_been_called = True

                        nonlocal original_result

                        original_result = original(*_og_args, **_og_kwargs)
                        return original_result

                    return call_original_fn_with_event_logging(
                        _original_fn, og_args, og_kwargs
                    )

                # Apply the name, docstring, and signature of `original` to `call_original`.
                # This is important because several autologging patch implementations inspect
                # the signature of the `original` argument during execution
                call_original = update_wrapper_extended(call_original, original)

                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_patch_function_start,
                    session,
                    destination,
                    function_name,
                    args,
                    kwargs,
                )

                if patch_is_class:
                    patch_function.call(call_original, *args, **kwargs)
                else:
                    patch_function(call_original, *args, **kwargs)

                session.state = "succeeded"

                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_patch_function_success,
                    session,
                    destination,
                    function_name,
                    args,
                    kwargs,
                )
            except Exception as e:
                session.state = "failed"
                patch_function_exception = e
                # Exceptions thrown during execution of the original function should be
                # propagated to the caller. Additionally, exceptions encountered during test
                # mode should be reraised to detect bugs in autologging implementations
                if failed_during_original or is_testing():
                    raise

            try:
                if original_has_been_called:
                    return original_result
                else:
                    return call_original_fn_with_event_logging(original, args, kwargs)
            finally:
                # If original function succeeds, but `patch_function_exception` exists,
                # it represent patching code unexpected failure, so we call
                # `log_patch_function_error` in this case.
                # If original function failed, we don't call `log_patch_function_error`
                # even if `patch_function_exception` exists, because original function failure
                # means there's some error in user code (e.g. user provide wrong arguments)
                if patch_function_exception is not None and not failed_during_original:
                    try_log_autologging_event(
                        AutologgingEventLogger.get_logger().log_patch_function_error,
                        session,
                        destination,
                        function_name,
                        args,
                        kwargs,
                        patch_function_exception,
                    )

                    _logger.warning(
                        "Encountered unexpected error during %s autologging: %s",
                        autologging_integration,
                        patch_function_exception,
                    )

    if is_property_method:
        # Create a patched function (also property decorated)
        # like:
        #
        # class A:
        # @property
        # def get_bound_safe_patch_fn(self):
        #   original_fn.fget(self) # do availability check
        #   return bound_safe_patch_fn
        #
        # Suppose `a1` is instance of class A,
        # then `a1.get_bound_safe_patch_fn(*args, **kwargs)` will be equivalent to
        # `bound_safe_patch_fn(*args, **kwargs)`
        def get_bound_safe_patch_fn(self):
            # This `original_fn.fget` call is for availability check, if it raise error
            # then `hasattr(obj, {func_name})` will return False
            # so it mimic the original property behavior.
            original_fn.fget(self)

            def bound_safe_patch_fn(*args, **kwargs):
                return safe_patch_function(self, *args, **kwargs)

            # Make bound method `instance.target_method` keep the same doc and signature
            bound_safe_patch_fn = update_wrapper_extended(
                bound_safe_patch_fn, original_fn.fget
            )
            # Here return the bound safe patch function because user call property decorated
            # method will like `instance.property_decorated_method(...)`, and internally it will
            # call the `bound_safe_patch_fn`, the argument list don't include the `self` argument,
            # so return bound function here.
            return bound_safe_patch_fn

        # Make unbound method `class.target_method` keep the same doc and signature
        get_bound_safe_patch_fn = update_wrapper_extended(
            get_bound_safe_patch_fn, original_fn.fget
        )
        safe_patch_obj = property(get_bound_safe_patch_fn)
    else:
        safe_patch_obj = update_wrapper_extended(safe_patch_function, original)

    new_patch = _wrap_patch(destination, function_name, safe_patch_obj)
    _store_patch(autologging_integration, new_patch)


# Represents an active autologging session using two fields:
# - integration: the name of the autologging integration corresponding to the session
# - id: a unique session identifier (e.g., a UUID)
# - state: the state of AutologgingSession, will be one of running/succeeded/failed
class AutologgingSession:
    def __init__(self, integration, id_):
        self.integration = integration
        self.id = id_
        self.state = "running"


class _AutologgingSessionManager:
    _session = None

    @classmethod
    @contextmanager
    def start_session(cls, integration):
        try:
            prev_session = cls._session
            if prev_session is None:
                session_id = uuid.uuid4().hex
                cls._session = AutologgingSession(integration, session_id)
            yield cls._session
        finally:
            # Only end the session upon termination of the context if we created
            # the session; otherwise, leave the session open for later termination
            # by its creator
            if prev_session is None:
                cls._end_session()

    @classmethod
    def active_session(cls):
        return cls._session

    @classmethod
    def _end_session(cls):
        cls._session = None


def update_wrapper_extended(wrapper, wrapped):
    """
    Update a `wrapper` function to look like the `wrapped` function. This is an extension of
    `functools.update_wrapper` that applies the docstring *and* signature of `wrapped` to
    `wrapper`, producing a new function.
    :return: A new function with the same implementation as `wrapper` and the same docstring
             & signature as `wrapped`.
    """
    updated_wrapper = functools.update_wrapper(wrapper, wrapped)
    # Assign the signature of the `wrapped` function to the updated wrapper function.
    # Certain frameworks may disallow signature inspection, causing `inspect.signature()` to throw.
    # One such example is the `tensorflow.estimator.Estimator.export_savedmodel()` function
    try:
        updated_wrapper.__signature__ = inspect.signature(wrapped)
    except Exception:
        _logger.debug(
            "Failed to restore original signature for wrapper around %s", wrapped
        )
    return updated_wrapper


def _wrap_patch(destination, name, patch_obj, settings=None):
    """
    Apply a patch.
    :param destination: Patch destination
    :param name: Name of the attribute at the destination
    :param patch_obj: Patch object, it should be a function or a property decorated function
                      to be assigned to the patch point {destination}.{name}
    :param settings: Settings for gorilla.Patch
    """
    if settings is None:
        settings = gorilla.Settings(allow_hit=True, store_hit=True)

    patch = gorilla.Patch(destination, name, patch_obj, settings=settings)
    gorilla.apply(patch)
    return patch


def _store_patch(autologging_integration, patch):
    """
    Stores a patch for a specified autologging_integration class. Later to be used for being able
    to revert the patch when disabling autologging.
    :param autologging_integration: The name of the autologging integration associated with the
                                    patch.
    :param patch: The patch to be stored.
    """
    if autologging_integration in _AUTOLOGGING_PATCHES:
        _AUTOLOGGING_PATCHES[autologging_integration].add(patch)
    else:
        _AUTOLOGGING_PATCHES[autologging_integration] = {patch}
