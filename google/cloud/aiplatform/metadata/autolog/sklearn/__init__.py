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


from typing import Optional, Dict, Union

import proto
import logging
from google.cloud import aiplatform
import sklearn
import numpy as np
from collections import defaultdict
import functools
from google.cloud.aiplatform.metadata.autolog.sklearn import utils as sklearn_utils
from google.cloud.aiplatform.metadata.autolog import (
    AutologgingQueue,
    _get_new_training_session_class,
    disable_autologging,
    safe_patch,
)
from google.cloud.aiplatform.metadata.metadata import _experiment_tracker

_logger = logging.Logger(__name__)
_SklearnTrainingSession = _get_new_training_session_class()

FLAVOR_NAME = "sklearn"

# The `_apis_autologging_disabled` contains APIs which is incompatible with autologging,
# when user call these APIs, autolog is temporarily disabled.
_apis_autologging_disabled = [
    "cross_validate",
    "cross_val_predict",
    "cross_val_score",
    "learning_curve",
    "permutation_test_score",
    "validation_curve",
]


class _AutologgingMetricsManager:
    """
    This class is designed for holding information which is used by autologging metrics
    It will hold information of:
    (1) a map of "prediction result object id" to a tuple of dataset name(the dataset is
       the one which generate the prediction result) and run_id.
       Note: We need this map instead of setting the run_id into the "prediction result object"
       because the object maybe a numpy array which does not support additional attribute
       assignment.
    (2) _log_post_training_metrics_enabled flag, in the following method scope:
       `model.fit`, `eval_and_log_metrics`, `model.score`,
       in order to avoid nested/duplicated autologging metric, when run into these scopes,
       we need temporarily disable the metric autologging.
    (3) _eval_dataset_info_map, it is a double level map:
       `_eval_dataset_info_map[run_id][eval_dataset_var_name]` will get a list, each
       element in the list is an id of "eval_dataset" instance.
       This data structure is used for:
        * generating unique dataset name key when autologging metric. For each eval dataset object,
          if they have the same eval_dataset_var_name, but object ids are different,
          then they will be assigned different name (via appending index to the
          eval_dataset_var_name) when autologging.
    (4) _metric_api_call_info, it is a double level map:
       `_metric_api_call_info[run_id][metric_name]` wil get a list of tuples, each tuple is:
         (logged_metric_key, metric_call_command_string)
        each call command string is like `metric_fn(arg1, arg2, ...)`
        This data structure is used for:
         * storing the call arguments dict for each metric call, we need log them into metric_info
           artifact file.
    Note: this class is not thread-safe.
    Design rule for this class:
     Because this class instance is a global instance, in order to prevent memory leak, it should
     only holds IDs and other small objects references. This class internal data structure should
     avoid reference to user dataset variables or model variables.
    """

    def __init__(self):
        self._pred_result_id_to_dataset_name_and_run_id = {}
        self._eval_dataset_info_map = defaultdict(lambda: defaultdict(list))
        self._metric_api_call_info = defaultdict(lambda: defaultdict(list))
        self._log_post_training_metrics_enabled = True
        self._metric_info_artifact_need_update = defaultdict(lambda: False)

    def should_log_post_training_metrics(self):
        """
        Check whether we should run patching code for autologging post training metrics.
        This checking should surround the whole patched code due to the safe guard checking,
        See following note.
        Note: It includes checking `_SklearnTrainingSession.is_active()`, This is a safe guarding
        for meta-estimator (e.g. GridSearchCV) case:
          running GridSearchCV.fit, the nested `estimator.fit` will be called in parallel,
          but, the _autolog_training_status is a global status without thread-safe lock protecting.
          This safe guarding will prevent code run into this case.
        """
        return (
            not _SklearnTrainingSession.is_active()
            and self._log_post_training_metrics_enabled
        )

    def disable_log_post_training_metrics(self):
        class LogPostTrainingMetricsDisabledScope:
            def __enter__(inner_self):  # pylint: disable=no-self-argument
                # pylint: disable=attribute-defined-outside-init
                inner_self.old_status = self._log_post_training_metrics_enabled
                self._log_post_training_metrics_enabled = False

            # pylint: disable=no-self-argument
            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                self._log_post_training_metrics_enabled = inner_self.old_status

        return LogPostTrainingMetricsDisabledScope()

    @staticmethod
    def get_run_name_for_model(model):
        return getattr(model, "_run_name", None)

    @staticmethod
    def is_metric_value_loggable(metric_value):
        """
        check whether the specified `metric_value` is a numeric value which can be logged
        as an MLflow metric.
        """
        return isinstance(metric_value, (int, float, np.number)) and not isinstance(
            metric_value, bool
        )

    def register_model(self, model, run_name):
        """
        In `patched_fit`, we need register the model with the run_id used in `patched_fit`
        So that in following metric autologging, the metric will be logged into the registered
        run_id
        """
        model._run_name = run_name

    @staticmethod
    def gen_name_with_index(name, index):
        assert index >= 0
        if index == 0:
            return name
        else:
            return f"{name}-{index + 1}"


_AUTOLOGGING_METRICS_MANAGER = _AutologgingMetricsManager()


def autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5,
    log_post_training_metrics=True,
    pos_label=None,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging for scikit-learn estimators.
    **When is autologging performed?**
      Autologging is performed when you call:
      - ``estimator.fit()``
      - ``estimator.fit_predict()``
      - ``estimator.fit_transform()``
    **Logged information**
      **Parameters**
        - Parameters obtained by ``estimator.get_params(deep=True)``. Note that ``get_params``
          is called with ``deep=True``. This means when you fit a meta estimator that chains
          a series of estimators, the parameters of these child estimators are also logged.
      **Training metrics**
        - A training score obtained by ``estimator.score``. Note that the training score is
          computed using parameters given to ``fit()``.
        - Common metrics for classifier:
          - `precision score`_
          .. _precision score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
          - `recall score`_
          .. _recall score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
          - `f1 score`_
          .. _f1 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
          - `accuracy score`_
          .. _accuracy score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
          If the classifier has method ``predict_proba``, we additionally log:
          - `log loss`_
          .. _log loss:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
          - `roc auc score`_
          .. _roc auc score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        - Common metrics for regressor:
          - `mean squared error`_
          .. _mean squared error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
          - root mean squared error
          - `mean absolute error`_
          .. _mean absolute error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
          - `r2 score`_
          .. _r2 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
      .. _post training metrics:
      **Post training metrics**
        When users call metric APIs after model training, MLflow tries to capture the metric API
        results and log them as MLflow metrics to the Run associated with the model. The following
        types of scikit-learn metric APIs are supported:
        - model.score
        - metric APIs defined in the `sklearn.metrics` module
        For post training metrics autologging, the metric key format is:
        "{metric_name}[-{call_index}]_{dataset_name}"
        - If the metric function is from `sklearn.metrics`, the MLflow "metric_name" is the
          metric function name. If the metric function is `model.score`, then "metric_name" is
          "{model_class_name}_score".
        - If multiple calls are made to the same scikit-learn metric API, each subsequent call
          adds a "call_index" (starting from 2) to the metric key.
        - MLflow uses the prediction input dataset variable name as the "dataset_name" in the
          metric key. The "prediction input dataset variable" refers to the variable which was
          used as the first argument of the associated `model.predict` or `model.score` call.
          Note: MLflow captures the "prediction input dataset" instance in the outermost call
          frame and fetches the variable name in the outermost call frame. If the "prediction
          input dataset" instance is an intermediate expression without a defined variable
          name, the dataset name is set to "unknown_dataset". If multiple "prediction input
          dataset" instances have the same variable name, then subsequent ones will append an
          index (starting from 2) to the inspected dataset name.
        **Limitations**
           - MLflow can only map the original prediction result object returned by a model
             prediction API (including predict / predict_proba / predict_log_proba / transform,
             but excluding fit_predict / fit_transform.) to an MLflow run.
             MLflow cannot find run information
             for other objects derived from a given prediction result (e.g. by copying or selecting
             a subset of the prediction result). scikit-learn metric APIs invoked on derived objects
             do not log metrics to MLflow.
           - Autologging must be enabled before scikit-learn metric APIs are imported from
             `sklearn.metrics`. Metric APIs imported before autologging is enabled do not log
             metrics to MLflow runs.
           - If user define a scorer which is not based on metric APIs in `sklearn.metrics`, then
             then post training metric autologging for the scorer is invalid.
        **Tags**
          - An estimator class name (e.g. "LinearRegression").
          - A fully qualified estimator class name
            (e.g. "sklearn.linear_model._base.LinearRegression").
        **Artifacts**
          - An MLflow Model with the :py:mod:`mlflow.sklearn` flavor containing a fitted estimator
            (logged by :py:func:`mlflow.sklearn.log_model()`). The Model also contains the
            :py:mod:`mlflow.pyfunc` flavor when the scikit-learn estimator defines `predict()`.
          - For post training metrics API calls, a "metric_info.json" artifact is logged. This is a
            JSON object whose keys are MLflow post training metric names
            (see "Post training metrics" section for the key format) and whose values are the
            corresponding metric call commands that produced the metrics, e.g.
            ``accuracy_score(y_true=test_iris_y, y_pred=pred_iris_y, normalize=False)``.
    **How does autologging work for meta estimators?**
      When a meta estimator (e.g. `Pipeline`_, `GridSearchCV`_) calls ``fit()``, it internally calls
      ``fit()`` on its child estimators. Autologging does NOT perform logging on these constituent
      ``fit()`` calls.
      **Parameter search**
          In addition to recording the information discussed above, autologging for parameter
          search meta estimators (`GridSearchCV`_ and `RandomizedSearchCV`_) records child runs
          with metrics for each set of explored parameters, as well as artifacts and parameters
          for the best model (if available).
    **Supported estimators**
      - All estimators obtained by `sklearn.utils.all_estimators`_ (including meta estimators).
      - `Pipeline`_
      - Parameter search estimators (`GridSearchCV`_ and `RandomizedSearchCV`_)
    .. _sklearn.utils.all_estimators:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.all_estimators.html
    .. _Pipeline:
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    .. _GridSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    .. _RandomizedSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    **Example**
    `See more examples <https://github.com/mlflow/mlflow/blob/master/examples/sklearn_autolog>`_
    .. code-block:: python
        from pprint import pprint
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import mlflow
        from mlflow import MlflowClient
        def fetch_logged_data(run_id):
            client = MlflowClient()
            data = client.get_run(run_id).data
            tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
            return data.params, data.metrics, tags, artifacts
        # enable autologging
        mlflow.sklearn.autolog()
        # prepare training data
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        # train a model
        model = LinearRegression()
        with mlflow.start_run() as run:
            model.fit(X, y)
        # fetch logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        pprint(params)
        # {'copy_X': 'True',
        #  'fit_intercept': 'True',
        #  'n_jobs': 'None',
        #  'normalize': 'False'}
        pprint(metrics)
        # {'training_score': 1.0,
           'training_mae': 2.220446049250313e-16,
           'training_mse': 1.9721522630525295e-31,
           'training_r2_score': 1.0,
           'training_rmse': 4.440892098500626e-16}
        pprint(tags)
        # {'estimator_class': 'sklearn.linear_model._base.LinearRegression',
        #  'estimator_name': 'LinearRegression'}
        pprint(artifacts)
        # ['model/MLmodel', 'model/conda.yaml', 'model/model.pkl']
    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with scikit-learn model artifacts during training. If
                               ``False``, input examples are not logged.
                               Note: Input examples are MLflow model attributes
                               and are only collected if ``log_models`` is also ``True``.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with scikit-learn model artifacts during training. If ``False``,
                                 signatures are not logged.
                                 Note: Model signatures are MLflow model attributes
                                 and are only collected if ``log_models`` is also ``True``.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
                       Input examples and model signatures, which are attributes of MLflow models,
                       are also omitted when ``log_models`` is ``False``.
    :param disable: If ``True``, disables the scikit-learn autologging integration. If ``False``,
                    enables the scikit-learn autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      scikit-learn that have not been tested against this version of the MLflow
                      client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during scikit-learn
                   autologging. If ``False``, show all events and warnings during scikit-learn
                   autologging.
    :param max_tuning_runs: The maximum number of child Mlflow runs created for hyperparameter
                            search estimators. To create child runs for the best `k` results from
                            the search, set `max_tuning_runs` to `k`. The default value is to track
                            the best 5 search parameter sets. If `max_tuning_runs=None`, then
                            a child run is created for each search parameter set. Note: The best k
                            results is based on ordering in `rank_test_score`. In the case of
                            multi-metric evaluation with a custom scorer, the first scorer’s
                            `rank_test_score_<scorer_name>` will be used to select the best k
                            results. To change metric used for selecting best k results, change
                            ordering of dict passed as `scoring` parameter for estimator.
    :param log_post_training_metrics: If ``True``, post training metrics are logged. Defaults to
                                      ``True``. See the `post training metrics`_ section for more
                                      details.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    :param pos_label: If given, used as the positive label to compute binary classification
                      training metrics such as precision, recall, f1, etc. This parameter should
                      only be set for binary classification model. If used for multi-label model,
                      the training metrics calculation will fail and the training metrics won't
                      be logged. If used for regression model, the parameter will be ignored.
    """
    _autolog(
        flavor_name=FLAVOR_NAME,
        log_post_training_metrics=log_post_training_metrics,
        pos_label=pos_label,
    )


def _autolog(
    flavor_name=FLAVOR_NAME,
    log_post_training_metrics=True,
    pos_label=None,
):
    """
    Internal autologging function for scikit-learn models.
    """

    def fit_vertex(original, self, *args, **kwargs):
        """
        Autologging function that performs model training by executing the training method
        referred to be `func_name` on the instance of `clazz` referred to by `self` & records
        MLflow parameters, metrics, tags, and artifacts to a corresponding MLflow Run.
        """
        # Obtain a copy of the training dataset prior to model training for subsequent
        # use during model logging & input example extraction, ensuring that we don't
        # attempt to infer input examples on data that was mutated during training
        (X, y_true, sample_weight) = sklearn_utils._get_X_y_and_sample_weight(
            self.fit, args, kwargs
        )
        autologging_queue = AutologgingQueue()
        if not _experiment_tracker.experiment_run:
            raise ValueError(
                "No experimentRun set. Make sure to call aiplatform.start_run('my-run') before training your model."
            )
        _log_pretraining_data(autologging_queue, self, *args, **kwargs)
        params_logging_future = autologging_queue.flush(synchronous=False)
        fit_output = original(self, *args, **kwargs)
        _log_posttraining_metadata(autologging_queue, self, X, y_true, sample_weight)
        autologging_queue.flush(synchronous=True)
        params_logging_future.await_completion()
        _experiment_tracker._experiment_run = aiplatform.ExperimentRun(
            run=_experiment_tracker.experiment_run.name,
            experiment=_experiment_tracker.experiment_name
        )
        return fit_output

    def _log_pretraining_data(
        autologging_queue, estimator, *args, **kwargs
    ):  # pylint: disable=unused-argument
        """
        Records metadata (e.g., params and tags) for a scikit-learn estimator prior to training.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.
        :param autologging_queue: An instance of `MlflowAutologgingQueueingClient` used for
                                   efficiently logging run data to MLflow Tracking.
        :param estimator: The scikit-learn estimator for which to log metadata.
        :param args: The arguments passed to the scikit-learn training routine (e.g.,
                     `fit()`, `fit_transform()`, ...).
        :param kwargs: The keyword arguments passed to the scikit-learn training routine.
        """

        # won't log deep params for search estimators
        deep = not sklearn_utils._is_search_estimator(estimator)
        autologging_queue.log_params(
            run_name=_experiment_tracker.experiment_run.name,
            params=estimator.get_params(deep=deep),
        )

    def _log_posttraining_metadata(autologging_queue, estimator, X, y, sample_weight):
        """
        Records metadata for a scikit-learn estimator after training has completed.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.
        :param autologging_queue: An instance of `MlflowAutologgingQueueingClient` used for
                                   efficiently logging run data to MLflow Tracking.
        :param estimator: The scikit-learn estimator for which to log metadata.
        :param X: The training dataset samples passed to the ``estimator.fit()`` function.
        :param y: The training dataset labels passed to the ``estimator.fit()`` function.
        :param sample_weight: Sample weights passed to the ``estimator.fit()`` function.
        """

        # log common metrics and artifacts for estimators (classifier, regressor)
        logged_metrics = sklearn_utils._log_estimator_content(
            autologging_queue=autologging_queue,
            estimator=estimator,
            prefix="training_",
            run_name=_experiment_tracker.experiment_run.name,
            X=X,
            y_true=y,
            sample_weight=sample_weight,
            pos_label=pos_label,
        )
        if y is None and not logged_metrics:
            _logger.warning(
                "Training metrics will not be recorded because training labels were not specified."
                " To automatically record training metrics, provide training labels as inputs to"
                " the model training function."
            )

        if sklearn_utils._is_search_estimator(estimator):
            if hasattr(estimator, "best_score_"):
                autologging_queue.log_metrics(
                    run_name=_experiment_tracker.experiment_run.name,
                    metrics={"best_cv_score": estimator.best_score_},
                )

    def patched_fit(fit_impl, original, self, *args, **kwargs):
        """
        Autologging patch function to be applied to a sklearn model class that defines a `fit`
        method and inherits from `BaseEstimator` (thereby defining the `get_params()` method)
        :param clazz: The scikit-learn model class to which this patch function is being applied for
                      autologging (e.g., `sklearn.linear_model.LogisticRegression`)
        :param func_name: The function name on the specified `clazz` that this patch is overriding
                          for autologging (e.g., specify "fit" in order to indicate that
                          `sklearn.linear_model.LogisticRegression.fit()` is being patched)
        """
        should_log_post_training_metrics = (
            log_post_training_metrics
            and _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics()
        )

        with _SklearnTrainingSession(clazz=self.__class__, allow_children=False) as t:
            if t.should_log():
                # In `fit_mlflow` call, it will also call metric API for computing training metrics
                # so we need temporarily disable the post_training_metrics patching.
                with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                    result = fit_impl(original, self, *args, **kwargs)
                if should_log_post_training_metrics:
                    _AUTOLOGGING_METRICS_MANAGER.register_model(
                        self, _experiment_tracker.experiment_name
                    )
                return result
            else:
                return original(self, *args, **kwargs)

    estimators_to_patch = sklearn_utils._gen_estimators_to_patch()
    patched_fit_impl = fit_vertex

    for class_def in estimators_to_patch:
        # Patch fitting methods
        for func_name in ["fit", "fit_transform", "fit_predict"]:
            sklearn_utils._patch_estimator_method_if_available(
                flavor_name,
                class_def,
                func_name,
                functools.partial(patched_fit, patched_fit_impl),
                manage_run=True,
            )

        # # Patch inference methods
        # for func_name in ["predict", "predict_proba", "transform", "predict_log_proba"]:
        #     sklearn_utils._patch_estimator_method_if_available(
        #         flavor_name,
        #         class_def,
        #         func_name,
        #         patched_predict,
        #         manage_run=False,
        #     )

        # # Patch scoring methods
        # sklearn_utils._patch_estimator_method_if_available(
        #     flavor_name,
        #     class_def,
        #     "score",
        #     patched_model_score,
        #     manage_run=False,
        # )

    def patched_fn_with_autolog_disabled(original, *args, **kwargs):
        with disable_autologging():
            return original(*args, **kwargs)

    for disable_autolog_func_name in _apis_autologging_disabled:
        safe_patch(
            flavor_name,
            sklearn.model_selection,
            disable_autolog_func_name,
            patched_fn_with_autolog_disabled,
            manage_run=False,
        )
