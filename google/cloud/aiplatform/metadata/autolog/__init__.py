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


import inspect
import logging
import contextlib
from google.cloud.aiplatform.metadata.autolog.safety import *
from google.cloud.aiplatform.metadata.autolog.events import *
from google.cloud.aiplatform.metadata.autolog.client import *


INPUT_EXAMPLE_SAMPLE_ROWS = 5
ENSURE_AUTOLOGGING_ENABLED_TEXT = (
    "please ensure that autologging is enabled before constructing the dataset."
)
AUTOLOGGING_CONF_KEY_IS_GLOBALLY_CONFIGURED = "globally_configured"
AUTOLOGGING_INTEGRATIONS = {}
FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY = {
    "fastai": ("fastai", "fastai"),
    "gluon": ("mxnet", "gluon"),
    "keras": ("keras", "keras"),
    "lightgbm": ("lightgbm", "lightgbm"),
    "statsmodels": ("statsmodels", "statsmodels"),
    "tensorflow": ("tensorflow", "tensorflow"),
    "xgboost": ("xgboost", "xgboost"),
    "sklearn": ("sklearn", "sklearn"),
    "pytorch": ("pytorch_lightning", "pytorch-lightning"),
    "pyspark.ml": ("pyspark", "spark"),
}



def get_autologging_config(flavor_name, config_key, default_value=None):
    """
    Returns a desired config value for a specified autologging integration.
    Returns `None` if specified `flavor_name` has no recorded configs.
    If `config_key` is not set on the config object, default value is returned.
    :param flavor_name: An autologging integration flavor name.
    :param config_key: The key for the desired config value.
    :param default_value: The default_value to return
    """
    config = AUTOLOGGING_INTEGRATIONS.get(flavor_name)
    if config is not None:
        return config.get(config_key, default_value)
    else:
        return default_value


def autologging_is_disabled(integration_name):
    """
    Returns a boolean flag of whether the autologging integration is disabled.
    :param integration_name: An autologging integration flavor name.
    """
    explicit_disabled = get_autologging_config(integration_name, "disable", True)
    if explicit_disabled:
        return True

    # if (
    #     integration_name in FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY
    #     and not is_flavor_supported_for_associated_package_versions(integration_name)
    # ):
    #     return get_autologging_config(
    #         integration_name, "disable_for_unsupported_versions", False
    #     )

    return False


@contextlib.contextmanager
def disable_autologging():
    """
    Context manager that temporarily disables autologging globally for all integrations upon
    entry and restores the previous autologging configuration upon exit.
    """
    global _AUTOLOGGING_GLOBALLY_DISABLED
    _AUTOLOGGING_GLOBALLY_DISABLED = True
    yield None
    _AUTOLOGGING_GLOBALLY_DISABLED = False


def _get_new_training_session_class():
    """
    Returns a session manager class for nested autologging runs.
    Examples
    --------
    >>> class Parent: pass
    >>> class Child: pass
    >>> class Grandchild: pass
    >>>
    >>> _TrainingSession = _get_new_training_session_class()
    >>> with _TrainingSession(Parent, False) as p:
    ...     with _SklearnTrainingSession(Child, True) as c:
    ...         with _SklearnTrainingSession(Grandchild, True) as g:
    ...             print(p.should_log(), c.should_log(), g.should_log())
    True False False
    >>>
    >>> with _TrainingSession(Parent, True) as p:
    ...     with _TrainingSession(Child, False) as c:
    ...         with _TrainingSession(Grandchild, True) as g:
    ...             print(p.should_log(), c.should_log(), g.should_log())
    True True False
    >>>
    >>> with _TrainingSession(Child, True) as c1:
    ...     with _TrainingSession(Child, True) as c2:
    ...             print(c1.should_log(), c2.should_log())
    True False
    """
    # NOTE: The current implementation doesn't guarantee thread-safety, but that's okay for now
    # because:
    # 1. We don't currently have any use cases for allow_children=True.
    # 2. The list append & pop operations are thread-safe, so we will always clear the session stack
    #    once all _TrainingSessions exit.
    class _TrainingSession:
        _session_stack = []

        def __init__(self, clazz, allow_children=True):
            """
            A session manager for nested autologging runs.
            :param clazz: A class object that this session originates from.
            :param allow_children: If True, allows autologging in child sessions.
                                   If False, disallows autologging in all descendant sessions.
            """
            self.allow_children = allow_children
            self.clazz = clazz
            self._parent = None

        def __enter__(self):
            if len(_TrainingSession._session_stack) > 0:
                self._parent = _TrainingSession._session_stack[-1]
                self.allow_children = (
                    _TrainingSession._session_stack[-1].allow_children
                    and self.allow_children
                )
            _TrainingSession._session_stack.append(self)
            return self

        def __exit__(self, tp, val, traceback):
            _TrainingSession._session_stack.pop()

        def should_log(self):
            """
            Returns True when at least one of the following conditions satisfies:
            1. This session is the root session.
            2. The parent session allows autologging and its class differs from this session's
               class.
            """
            return (self._parent is None) or (
                self._parent.allow_children and self._parent.clazz != self.clazz
            )

        @staticmethod
        def is_active():
            return len(_TrainingSession._session_stack) != 0

    return _TrainingSession
