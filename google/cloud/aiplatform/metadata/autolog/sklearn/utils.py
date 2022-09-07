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

import collections
import inspect
import logging
import numpy as np
from copy import deepcopy
from packaging.version import Version
import sklearn

from google.cloud.aiplatform.metadata.autolog import gorilla
from google.cloud.aiplatform.metadata.autolog.safety import safe_patch


_logger = logging.getLogger(__name__)

_SAMPLE_WEIGHT = "sample_weight"

_SklearnMetric = collections.namedtuple(
    "_SklearnMetric", ["name", "function", "arguments"]
)


def _gen_estimators_to_patch():

    _, estimators_to_patch = zip(*sklearn.utils.all_estimators())

    # Include relevant meta estimators
    meta_estimators_to_patch = [
        sklearn.model_selection.GridSearchCV,
        sklearn.model_selection.RandomizedSearchCV,
        sklearn.pipeline.Pipeline,
    ]
    estimators_to_patch = set(estimators_to_patch).union(set(meta_estimators_to_patch))

    # Exclude certain preprocessing & feature manipulation estimators
    excluded_estimators = [sklearn.compose._column_transformer.ColumnTransformer]
    estimators_to_patch = estimators_to_patch.difference(set(excluded_estimators))

    excluded_module_names = [
        "sklearn.preprocessing",
        "sklearn.impute",
        "sklearn.feature_extraction",
        "sklearn.feature_selection",
    ]

    return [
        estimator
        for estimator in estimators_to_patch
        if not any(
            estimator.__module__.startswith(module_name)
            for module_name in excluded_module_names
        )
    ]


## Copy directly from MLflow
def _get_X_y_and_sample_weight(fit_func, fit_args, fit_kwargs):
    """
    Get a tuple of (X, y, sample_weight) in the following steps.
    1. Extract X and y from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in fit_func,
       extract it from fit_args or fit_kwargs and return (X, y, sample_weight),
       otherwise return (X, y)
    :param fit_func: A fit function object.
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :returns: A tuple of either (X, y, sample_weight), where `y` and `sample_weight` may be
              `None` if the specified `fit_args` and `fit_kwargs` do not specify labels or
              a sample weighting. Copies of `X` and `y` are made in order to avoid mutation
              of the dataset during training.
    """

    def _get_Xy(args, kwargs, X_var_name, y_var_name):
        # corresponds to: model.fit(X, y)
        if len(args) >= 2:
            return args[:2]

        # corresponds to: model.fit(X, <y_var_name>=y)
        if len(args) == 1:
            return args[0], kwargs.get(y_var_name)

        # corresponds to: model.fit(<X_var_name>=X, <y_var_name>=y)
        return kwargs[X_var_name], kwargs.get(y_var_name)

    def _get_sample_weight(arg_names, args, kwargs):
        sample_weight_index = arg_names.index(_SAMPLE_WEIGHT)

        # corresponds to: model.fit(X, y, ..., sample_weight)
        if len(args) > sample_weight_index:
            return args[sample_weight_index]

        # corresponds to: model.fit(X, y, ..., sample_weight=sample_weight)
        if _SAMPLE_WEIGHT in kwargs:
            return kwargs[_SAMPLE_WEIGHT]

        return None

    def _get_arg_names(f):
        return list(inspect.signature(f).parameters.keys())

    fit_arg_names = _get_arg_names(fit_func)
    # In most cases, X_var_name and y_var_name become "X" and "y", respectively.
    # However, certain sklearn models use different variable names for X and y.
    # E.g., see: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier.fit
    X_var_name, y_var_name = fit_arg_names[:2]
    X, y = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)
    if X is not None:
        X = deepcopy(X)
    if y is not None:
        y = deepcopy(y)
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if (_SAMPLE_WEIGHT in fit_arg_names)
        else None
    )

    return (X, y, sample_weight)


def _is_search_estimator(estimator):
    parameter_search_estimators = [
        sklearn.model_selection.GridSearchCV,
        sklearn.model_selection.RandomizedSearchCV,
    ]

    return any(
        isinstance(estimator, param_search_estimator)
        for param_search_estimator in parameter_search_estimators
    )


def _patch_estimator_method_if_available(
    flavor_name, class_def, func_name, patched_fn, manage_run
):
    if not hasattr(class_def, func_name):
        return

    original = gorilla.get_original_attribute(
        class_def, func_name, bypass_descriptor_protocol=False
    )
    # Retrieve raw attribute while bypassing the descriptor protocol
    raw_original_obj = gorilla.get_original_attribute(
        class_def, func_name, bypass_descriptor_protocol=True
    )
    if raw_original_obj == original and (
        callable(original) or isinstance(original, property)
    ):
        # normal method or property decorated method
        safe_patch(flavor_name, class_def, func_name, patched_fn, manage_run=manage_run)
    elif hasattr(raw_original_obj, "delegate_names") or hasattr(
        raw_original_obj, "check"
    ):
        # sklearn delegated method
        safe_patch(
            flavor_name, raw_original_obj, "fn", patched_fn, manage_run=manage_run
        )
    else:
        # unsupported method type. skip patching
        pass


def _log_specialized_estimator_content(
    autologging_queue,
    fitted_estimator,
    run_name,
    prefix,
    X,
    y_true,
    sample_weight,
    pos_label,
):
    import sklearn

    metrics = dict()

    if y_true is not None:
        try:
            if sklearn.base.is_classifier(fitted_estimator):
                metrics = _get_classifier_metrics(
                    fitted_estimator, prefix, X, y_true, sample_weight, pos_label
                )
            elif sklearn.base.is_regressor(fitted_estimator):
                metrics = _get_regressor_metrics(
                    fitted_estimator, prefix, X, y_true, sample_weight
                )
        except Exception as err:
            msg = (
                "Failed to autolog metrics for "
                + fitted_estimator.__class__.__name__
                + ". Logging error: "
                + str(err)
            )
            _logger.warning(msg)
        else:
            autologging_queue.log_metrics(run_name=run_name, metrics=metrics)

    return metrics


def _log_estimator_content(
    autologging_queue,
    estimator,
    run_name,
    prefix,
    X,
    y_true=None,
    sample_weight=None,
    pos_label=None,
):
    """
    Logs content for the given estimator, which includes metrics and artifacts that might be
    tailored to the estimator's type (e.g., regression vs classification). Training labels
    are required for metric computation; metrics will be omitted if labels are not available.
    :param autologging_queue: An instance of `MlflowAutologgingQueueingClient` used for
                               efficiently logging run data to MLflow Tracking.
    :param estimator: The estimator used to compute metrics and artifacts.
    :param run_id: The run under which the content is logged.
    :param prefix: A prefix used to name the logged content. Typically it's 'training_' for
                   training-time content and user-controlled for evaluation-time content.
    :param X: The data samples.
    :param y_true: Labels.
    :param sample_weight: Per-sample weights used in the computation of metrics and artifacts.
    :param pos_label: The positive label used to compute binary classification metrics such as
        precision, recall, f1, etc. This parameter is only used for classification metrics.
        If set to `None`, the function will calculate metrics for each label and find their
        average weighted by support (number of true instances for each label).
    :return: A dict of the computed metrics.
    """
    metrics = _log_specialized_estimator_content(
        autologging_queue=autologging_queue,
        fitted_estimator=estimator,
        run_name=run_name,
        prefix=prefix,
        X=X,
        y_true=y_true,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )

    if hasattr(estimator, "score") and y_true is not None:
        try:
            # Use the sample weight only if it is present in the score args
            score_arg_names = _get_arg_names(estimator.score)
            score_args = (
                (X, y_true, sample_weight)
                if _SAMPLE_WEIGHT in score_arg_names
                else (X, y_true)
            )
            score = estimator.score(*score_args)
        except Exception as e:
            msg = (
                estimator.score.__qualname__
                + " failed. The 'training_score' metric will not be recorded. Scoring error: "
                + str(e)
            )
            _logger.warning(msg)
        else:
            score_key = prefix + "score"
            autologging_queue.log_metrics(
                run_name=run_name, metrics={score_key: score}
            )
            metrics[score_key] = score

    return metrics


def _log_warning_for_metrics(func_name, func_call, err):
    msg = (
        func_call.__qualname__
        + " failed. The metric "
        + func_name
        + " will not be recorded."
        + " Metric error: "
        + str(err)
    )
    _logger.warning(msg)


def _get_metrics_value_dict(metrics_list):
    metric_value_dict = {}
    for metric in metrics_list:
        try:
            metric_value = metric.function(**metric.arguments)
        except Exception as e:
            _log_warning_for_metrics(metric.name, metric.function, e)
        else:
            metric_value_dict[metric.name] = metric_value
    return metric_value_dict


def _get_classifier_metrics(
    fitted_estimator, prefix, X, y_true, sample_weight, pos_label
):
    """
    Compute and record various common metrics for classifiers
    For (1) precision score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    (2) recall score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    (3) f1_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    By default, when `pos_label` is not specified (passed in as `None`), we set `average`
    to `weighted` to compute the weighted score of these metrics.
    When the `pos_label` is specified (not `None`), we set `average` to `binary`.
    For (4) accuracy score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    we choose the parameter `normalize` to be `True` to output the percentage of accuracy,
    as opposed to `False` that outputs the absolute correct number of sample prediction
    We log additional metrics if certain classifier has method `predict_proba`
    (5) log loss:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    (6) roc_auc_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    By default, for roc_auc_score, we pick `average` to be `weighted`, `multi_class` to be `ovo`,
    to make the output more insensitive to dataset imbalance.
    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, ...... sample_weight), otherwise as (y_true, y_pred, ......)
    3. return a dictionary of metric(name, value)
    :param fitted_estimator: The already fitted classifier
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """

    average = "weighted" if pos_label is None else "binary"
    y_pred = fitted_estimator.predict(X)

    classifier_metrics = [
        _SklearnMetric(
            name=prefix + "precision_score",
            function=sklearn.metrics.precision_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
        ),
        _SklearnMetric(
            name=prefix + "recall_score",
            function=sklearn.metrics.recall_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
        ),
        _SklearnMetric(
            name=prefix + "f1_score",
            function=sklearn.metrics.f1_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
        ),
        _SklearnMetric(
            name=prefix + "accuracy_score",
            function=sklearn.metrics.accuracy_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                normalize=True,
                sample_weight=sample_weight,
            ),
        ),
    ]

    if hasattr(fitted_estimator, "predict_proba"):
        y_pred_proba = fitted_estimator.predict_proba(X)
        classifier_metrics.extend(
            [
                _SklearnMetric(
                    name=prefix + "log_loss",
                    function=sklearn.metrics.log_loss,
                    arguments=dict(
                        y_true=y_true, y_pred=y_pred_proba, sample_weight=sample_weight
                    ),
                ),
            ]
        )

        if _is_metric_supported("roc_auc_score"):
            # For binary case, the parameter `y_score` expect scores must be
            # the scores of the class with the greater label.
            if len(y_pred_proba[0]) == 2:
                y_pred_proba = y_pred_proba[:, 1]

            classifier_metrics.extend(
                [
                    _SklearnMetric(
                        name=prefix + "roc_auc_score",
                        function=sklearn.metrics.roc_auc_score,
                        arguments=dict(
                            y_true=y_true,
                            y_score=y_pred_proba,
                            average="weighted",
                            sample_weight=sample_weight,
                            multi_class="ovo",
                        ),
                    ),
                ]
            )

    return _get_metrics_value_dict(classifier_metrics)


def _get_regressor_metrics(fitted_estimator, prefix, X, y_true, sample_weight):
    """
    Compute and record various common metrics for regressors
    For (1) (root) mean squared error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    (2) mean absolute error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    (3) r2 score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    By default, we choose the parameter `multioutput` to be `uniform_average`
    to average outputs with uniform weight.
    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, sample_weight, multioutput), otherwise as (y_true, y_pred, multioutput)
    3. return a dictionary of metric(name, value)
    :param fitted_estimator: The already fitted regressor
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    y_pred = fitted_estimator.predict(X)

    regressor_metrics = [
        _SklearnMetric(
            name=prefix + "mse",
            function=sklearn.metrics.mean_squared_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=prefix + "mae",
            function=sklearn.metrics.mean_absolute_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=prefix + "r2_score",
            function=sklearn.metrics.r2_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
    ]

    # To be compatible with older versions of scikit-learn (below 0.22.2), where
    # `sklearn.metrics.mean_squared_error` does not have "squared" parameter to calculate `rmse`,
    # we compute it through np.sqrt(<value of mse>)
    metrics_value_dict = _get_metrics_value_dict(regressor_metrics)
    metrics_value_dict[prefix + "rmse"] = np.sqrt(metrics_value_dict[prefix + "mse"])

    return metrics_value_dict


def _is_metric_supported(metric_name):
    """Util function to check whether a metric is able to be computed in given sklearn version"""

    # This dict can be extended to store special metrics' specific supported versions
    _metric_supported_version = {"roc_auc_score": "0.22.2"}

    return Version(sklearn.__version__) >= Version(
        _metric_supported_version[metric_name]
    )


def _get_arg_names(f):
    """
    Get the argument names of a function.
    :param f: A function.
    :return: A list of argument names.
    """
    # `inspect.getargspec` or `inspect.getfullargspec` doesn't work properly for a wrapped function.
    # See https://hynek.me/articles/decorators#mangled-signatures for details.
    return list(inspect.signature(f).parameters.keys())
