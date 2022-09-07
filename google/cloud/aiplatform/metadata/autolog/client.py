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
import logging
from datetime import datetime
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

from google.cloud import aiplatform
from google.cloud.aiplatform.metadata.metadata import _experiment_tracker


_logger = logging.getLogger(__name__)

_PendingCreateRun = namedtuple("_PendingCreateRun", ["run_name"])
_PendingEndRun = namedtuple("_PendingEndRun", [])


class PendingRunName:
    """
    Serves as a placeholder for the run name that does not yet exist, enabling additional
    metadata (e.g. metrics, params, ...) to be enqueued for the run prior to its creation.
    """


class RunOperations:
    """
    Represents a collection of operations on one or more ExperimentRuns, such as run creation
    or metric logging.
    """

    def __init__(self, operation_futures):
        self._operation_futures = operation_futures

    def await_completion(self):
        """
        Blocks on completion of the MLflow Run operations.
        """
        failed_operations = []
        for future in self._operation_futures:
            try:
                future.result()
            except Exception as e:
                failed_operations.append(e)

        if len(failed_operations) > 0:
            raise Exception(
                f"The following failures occurred while performing one or more logging operations: {failed_operations}"
            )


class _PendingRunOperations:
    """
    Represents a collection of queued / pending  ExperimentRun operations.
    """

    def __init__(self, run):
        self.run = run
        self.create_run = None
        self.end_run = None
        self.params_queue = {}
        self.metrics_queue = {}

    def enqueue(self, params=None, metrics=None, create_run=None, end_run=None):
        """
        Enqueues a new pending logging operation for the associated ExperimentRun.
        """
        if create_run:
            assert (
                not self.create_run
            ), "Attempted to create the same run multiple times"
            self.create_run = create_run
        if end_run:
            assert not self.end_run, "Attempted to end the same run multiple times"
            self.end_run = end_run

        self.params_queue.update(params or {})
        self.metrics_queue.update(metrics or {})


# Define a threadpool for use across `MlflowAutologgingQueueingClient` instances to ensure that
# `MlflowAutologgingQueueingClient` instances can be pickled (ThreadPoolExecutor objects are not
# pickleable and therefore cannot be assigned as instance attributes).
#
# We limit the number of threads used for run operations, using at most 8 threads or 2 * the number
# of CPU cores available on the system (whichever is smaller)
num_cpus = os.cpu_count() or 4
num_logging_workers = min(num_cpus * 2, 8)
_AUTOLOGGING_QUEUE_THREAD_POOL = ThreadPoolExecutor(max_workers=num_logging_workers)


class AutologgingQueue:
    """
    Efficiently implements a subset of MLflow Tracking's  `MlflowClient` and fluent APIs to provide
    automatic batching and async execution of run operations by way of queueing, as well as
    parameter / tag truncation for autologging use cases. Run operations defined by this client,
    such as `create_run` and `log_metrics`, enqueue data for future persistence to MLflow
    Tracking. Data is not persisted until the queue is flushed via the `flush()` method, which
    supports synchronous and asynchronous execution.
    MlflowAutologgingQueueingClient is not threadsafe; none of its APIs should be called
    concurrently.
    """

    def __init__(self):
        self._pending_operations = {}
        if _experiment_tracker.experiment_name:
            self.experiment_name = _experiment_tracker.experiment_name
        else:
            raise ValueError(
                "No experiment set. Make sure to call aiplatform.init(experiment='my-experiment') before autologging."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):  # pylint: disable=unused-argument

        if exc is None and exc_type is None and traceback is None:
            self.flush(synchronous=True)
        else:
            _logger.debug(
                "Skipping run content logging upon AutologgingQueue because"
                " an exception was raised within the context: %s",
                exc,
            )

    def create_run(
        self,
    ) -> PendingRunName:
        """Enqueues a CreateRun operation.

        Return: A `PendingRunName` that can be passed as the `run_name` parameter to other client
                 logging APIs, such as `log_params` and `log_metrics`.
        """

        run_name = PendingRunName()
        self._get_pending_operations(run_name).enqueue(
            create_run=_PendingCreateRun(
                run_name=run_name,
            )
        )
        return run_name

    def end_run(
        self,
        run_name: Union[str, PendingRunName],
    ) -> None:
        """
        Enqueues a endRun operation for the specified `run_name`.
        """
        self._get_pending_operations(run_name).enqueue(end_run=_PendingEndRun())

    def log_params(
        self,
        run_name: Union[str, PendingRunName],
        params: Dict[str, Union[float, int, str]],
    ) -> None:
        """
        Enqueues a collection of Parameters to be logged to the run specified by `run_name`.
        """
        params = {
            k: v
            if (
                not isinstance(v, bool) and (isinstance(v, int) or isinstance(v, float))
            )
            else str(v)
            for k, v in params.items()
        }
        self._get_pending_operations(run_name).enqueue(params=params)

    def log_metrics(
        self,
        run_name: Union[str, PendingRunName],
        metrics: Dict[str, Union[float, int, str]],
    ) -> None:
        """
        Enqueues a collection of Metrics to be logged to the run specified by `run_name`.
        """
        self._get_pending_operations(run_name).enqueue(metrics=metrics)

    ## Todo: support log_time_series

    def _get_pending_operations(self, run_name):
        """
        :return: A `_PendingRunOperations` containing all pending operations for the
                 specified `run_id`.
        """
        if run_name not in self._pending_operations:
            self._pending_operations[run_name] = _PendingRunOperations(
                run=aiplatform.ExperimentRun(run_name, self.experiment_name),
            )
        return self._pending_operations[run_name]

    def _try_operation(self, fn, *args, **kwargs):
        """
        Attempt to evaluate the specified function, `fn`, on the specified `*args` and `**kwargs`,
        returning either the result of the function evaluation (if evaluation was successful) or
        the exception raised by the function evaluation (if evaluation was unsuccessful).
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

    def flush(self, synchronous=True):
        """
        Flushes all queued run operations, resulting in the creation or mutation of runs
        and run data.
        :param synchronous: If `True`, run operations are performed synchronously, and a
                            `RunOperations` result object is only returned once all operations
                            are complete. If `False`, run operations are performed asynchronously,
                            and an `RunOperations` object is returned that represents the ongoing
                            run operations.
        :return: A `RunOperations` instance representing the flushed operations. These operations
                 are already complete if `synchronous` is `True`. If `synchronous` is `False`, these
                 operations may still be inflight. Operation completion can be synchronously waited
                 on via `RunOperations.await_completion()`.
        """
        logging_futures = []
        for pending_operations in self._pending_operations.values():
            future = _AUTOLOGGING_QUEUE_THREAD_POOL.submit(
                self._flush_pending_operations,
                pending_operations=pending_operations,
            )
            logging_futures.append(future)
        self._pending_operations = {}

        logging_operations = RunOperations(logging_futures)
        if synchronous:
            logging_operations.await_completion()
        return logging_operations

    def _flush_pending_operations(self, pending_operations):
        """
        Synchronously and sequentially flushes the specified list of pending run operations.
        NB: Operations are not parallelized on a per-run basis because MLflow's File Store, which
        is frequently used for local ML development, does not support threadsafe metadata logging
        within a given run.
        """
        if pending_operations.create_run:
            TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
            run_name = f"sklearn-autologging-run-{TIMESTAMP}"
            run = aiplatform.ExperimentRun.create(run_name, experiment=self._experiment)
            pending_operations.run = run

        operation_results = []
        if pending_operations.params_queue:
            operation_results.append(
                self._try_operation(
                    pending_operations.run.log_params,
                    params=pending_operations.params_queue,
                )
            )
        if pending_operations.metrics_queue:
            operation_results.append(
                self._try_operation(
                    pending_operations.run.log_metrics,
                    metrics=pending_operations.metrics_queue,
                )
            )
        if pending_operations.end_run:
            operation_results.append(
                self._try_operation(
                    pending_operations.run.end_run,
                )
            )

        failures = [
            result for result in operation_results if isinstance(result, Exception)
        ]
        if len(failures) > 0:
            raise Exception(
                f"Failed to perform one or more operations on the run {pending_operations.run_name}. "
                + f"Failed operations: {failures}"
            )


__all__ = [
    "AutologgingQueue",
]
