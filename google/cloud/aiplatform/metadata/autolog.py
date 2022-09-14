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
from google.cloud import aiplatform


class RunInfo:
    def __init__(self):
        self.run = aiplatform.metadata.metadata._experiment_tracker.experiment_run
        if self.run:
            self.run_id = self.run.name
        else:
            self.run_id = None


class VertexActiveRun:
    def __init__(self):
        self.info = RunInfo()


def patch_active_run():
    return VertexActiveRun()


def patch_client_log_batch(client, run_id, metrics=[], params=[], tags=[]):
    # pylint: disable=unused-argument
    aiplatform.start_run(run_id, resume=True)
    params = {param.key: param.value for param in params}
    aiplatform.log_params(params)
    for metric in metrics:
        if metric.step:
            aiplatform.log_time_series_metrics({metric.key: metric.value}, metric.step)
        else:
            aiplatform.log_metrics({metric.key: metric.value})


def patch_client_log_metric(client, run_id, key, value, step=None, wall_time=None):
    # pylint: disable=unused-argument
    aiplatform.start_run(run_id, resume=True)
    if not step:
        aiplatform.log_metrics({key: value})
    else:
        aiplatform.log_time_series_metrics({key: value}, step)


def patch_client_log_param(client, run_id, key, value):
    # pylint: disable=unused-argument
    aiplatform.start_run(run_id, resume=True)
    aiplatform.log_params({key: value})


def patch_client_log_dict(client, run_id, dictionary, artifact_file):
    # pylint: disable=unused-argument
    ## Todo: requires design for logging mlflow style artifacts
    return


def patch_client_log_artifacts(client, run_id, local_path, artifact_path=None):
    # pylint: disable=unused-argument
    ## Todo: requires design for logging mlflow style artifacts
    return


def patch_log_model(**kwargs):
    # pylint: disable=unused-argument
    ## Todo: requires design for model serialization
    return


def patch_log_param(key, value):
    aiplatform.log_params({key: value})


def patch_log_params(params):
    aiplatform.log_params(params)


def patch_log_artifact(local_path, artifact_path=None):
    # pylint: disable=unused-argument
    ## Todo: requires design for logging mlflow style artifacts
    return


def autolog(
    log_input_examples: bool = False,
    log_model_signatures: bool = True,
    log_models: bool = True,
    disable: bool = False,
    exclusive: bool = False,
    disable_for_unsupported_versions: bool = False,
    silent: bool = False,
    # pylint: disable=unused-argument
) -> None:
    import mlflow

    mlflow.active_run = patch_active_run
    mlflow.MlflowClient.log_batch = patch_client_log_batch
    mlflow.MlflowClient.log_metric = patch_client_log_metric
    mlflow.MlflowClient.log_param = patch_client_log_param
    mlflow.MlflowClient.log_dict = patch_client_log_dict
    mlflow.MlflowClient.log_artifacts = patch_client_log_artifacts
    mlflow.models.Model.log = patch_log_model
    mlflow.log_param = patch_log_param
    mlflow.log_params = patch_log_params
    mlflow.log_artifact = patch_log_artifact

    mlflow.autolog(
        log_input_examples=log_input_examples,
        log_model_signatures=log_model_signatures,
        log_models=log_models,
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
    )
