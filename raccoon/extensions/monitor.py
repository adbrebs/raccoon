from collections import defaultdict

from raccoon.utils import str_grey
from raccoon.extensions.base import Extension


class MetricMonitor(Extension):
    """Extension to monitor metrics during training.

    This is an abstract class with two children classes:
        - `TrainMonitor`: the extension responsible for training and monitoring metrics on
            the training dataset.
        - `ValidationMonitor`: an extension for monitoring metrics on a validation dataset.

    Attributes:
        fun_batch_metrics (function):
            Takes as input a dictionary {'input_name': numpy array} and returns a dictionary of
            {'metric_name': scalar value}.
            the function to be called at each minibatch, taking as input the
            output of the data generator and returning some metrics.
            If one of the output of the generator has key name 'metric_counter', then this value
            is used to average minibatch values of the metrics. This is useful when
            minibatches have different lengths.
        metric_names (list of strings):
            List of metric names that the user wants to display during training.
            This list might be a subset only of the dict keys returned by `fun_batch_metrics`.

    Note:
        If you have minibatches of different sizes, fun_batch_metrics should return a dict with
        the key 'metric_counter' that returns the size of the current minibatch. This allows
        raccoon to compute an accurate average of the metrics.
    """

    def __init__(self, name, fun_batch_metrics, metric_names, freq, **kw):
        super().__init__(name, freq, **kw)

        self.fun_batch_metrics = fun_batch_metrics
        self.metric_names = metric_names

        # Stores all the values of the monitored metrics.
        self.history = defaultdict(list)
        self.iterations = defaultdict(list)

        self.checkpoint_attributes = ("history", "iterations")

    def _execute(self, batch_id, epoch_id, end_epoch=False):
        dict_metric_values = self.compute_metrics()

        # Save metrics in history
        for metric_name, v in dict_metric_values.items():
            self.history[metric_name].append(v)
            self.iterations[metric_name].append(batch_id)

        return self.log([f'{str_grey(metric)}: {dict_metric_values[metric]:.7g}'
                         for metric in self.metric_names])

    def compute_metrics(self):
        """Computes and returns a dictionary {'metric_name': value) computed with
        fun_batch_metrics. Called by the `execute` method.

        Returns:
            dict: Dictionary {'metric_name': numpy array) computed with fun_batch_metrics.
        """
        raise NotImplementedError


class ValidationMonitor(MetricMonitor):
    """Extension to monitor metrics computed on a validation dataset.

    Attributes:
        data_generator (generator function): generator yielding a dictionary {'input_name': numpy
            array corresponding to a minibatch}
            It iterates over a validation dataset and its output is fed to the `fun_batch_metrics`
            function.
    """

    def __init__(self, name, fun_batch_metrics, metric_names, freq,
                 data_generator, on_end=True, on_start=False):
        super().__init__(
            name, fun_batch_metrics, metric_names, freq,
            on_end=on_end, on_start=on_start)
        self.data_generator = data_generator

    def compute_metrics(self):

        dict_metrics = defaultdict(int)

        n_data = 0
        for data in self.data_generator():
            dict_values = self.fun_batch_metrics(data)
            bs = dict_values.get('metric_counter', 1)
            n_data += bs

            for n in self.metric_names:
                dict_metrics[n] += dict_values[n] * bs

        return {n: dict_metrics[n] / n_data for n in self.metric_names}


ValidMonitor = ValidationMonitor


class TrainMonitor(MetricMonitor):
    """
    Extension required by `class:Trainer` to train the model and monitor the
    metrics computed on the trainig dataset.
    """

    def __init__(self, fun_batch_metrics, metric_names, freq, on_end=False):
        super().__init__('Training', fun_batch_metrics, metric_names, freq,
                         on_start=False, on_end=on_end)

        # This extension updates the following variables while the train method is called:
        # Computed metrics until they are displayed
        self.current_metric_values = defaultdict(int)
        # Sum of the batch sizes (so that metrics can be accurately averaged)
        self.current_cumsum_batch_sizes = 0

    def compute_metrics(self):

        dict_metrics = {k: v / self.current_cumsum_batch_sizes
                        for k, v in self.current_metric_values.items()}

        # Reset current metric values for next pass
        self.current_metric_values = defaultdict(int)
        self.current_cumsum_batch_sizes = 0

        return dict_metrics

    def train(self, inputs):
        m_values = self.fun_batch_metrics(inputs)
        bs = m_values.get('metric_counter', 1)

        for n in self.metric_names:
            self.current_metric_values[n] += m_values[n] * bs

        self.current_cumsum_batch_sizes += bs
