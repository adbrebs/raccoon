import math
import time

from raccoon.extensions.base import Extension


class ValidationSchedule(Extension):
    """Extension that performs an action if there is no improvement on a metric monitored by a
    monitoring extension.
    If the metric does not improve for absolute_patience, then the training stops.

    The frequence of this extension is the same as the :class:`Monitor` monitor
    parameter.

    Attributes:
        monitor (:class:`Monitor`):
            :class:`Monitor` object which computes the metric you are
            interested in.
        metric_name (string):
            the name of the metric that you are interested in.
        metric_mode (string, either 'min' or 'max', default='min'):
            indicates if the metric should be minimized of maximized
        process_function (function with no arguments returning a string):
            the function to be called if the metric is not improved. If the function returns
            a string, it will be displayed.
            string will be displayed
        patience (int, default 5):
            the number of times we allow the metric to not improve before
            calling the process_function
        max_patience (int, default=7)
            the number of times we allow the metric to not improve before we
            stop the training.
        name (string, default None)
            Name of the extension.
    """

    def __init__(self, monitor, metric_name, process_function,
                 patience=5, max_patience=7, metric_mode='min',
                 name='Validation schedule'):
        super().__init__(name, monitor.freq)
        self.process_function = process_function
        self.patience = patience
        self.absolute_patience = max_patience

        self.validation_monitor = monitor

        self.metric_name = metric_name
        self.mode_metric = metric_mode
        if metric_mode == 'max':
            self.m = -1
        elif metric_mode == 'min':
            self.m = 1
        else:
            raise ValueError

        # The current number of times with no improvement since the last time process_function
        # was called.
        self.waiting = 0
        # The current total number of times with no improvement since last improvement.
        self.total_waiting = 0
        # The best value of the metric so far.
        self.best_value = self.m * float('Inf')

        self.checkpoint_attributes = ("waiting", "total_waiting", "best_value")

    def _execute(self, trainer=None, end_epoch=False):

        if self.total_waiting > self.absolute_patience:
            return self.log(['Patience exceeded'], stop_training=True)

        lines = []

        current_value = self.validation_monitor.history[self.metric_name][-1]

        if current_value * self.m < self.best_value * self.m:
            self.waiting = self.total_waiting = 0
            self.best_value = current_value
        else:
            self.waiting += 1
            self.total_waiting += 1

            if self.waiting > self.patience:
                msg = self.process_function()
                self.waiting = 0
                if msg:
                    lines.append(msg)

        lines.append(f'{self.display_info()} '
                     f'{self.waiting}/{self.patience}, '
                     f'total_waiting {self.total_waiting}/{self.absolute_patience}, '
                     f'best {self.metric_name} = {self.best_value:5g}')

        return self.log(lines)

    def display_info(self):
        return ''


class MutableScalarInterface:
    """This class handles the reading/writing of a mutable scalar"""

    def __init__(self, name):
        self.name = name

    def read(self):
        """Returns a scalar numpy build-in value"""
        raise NotImplementedError

    def write(self, scalar_value):
        """scalar_value: python built-in type: int, float"""
        raise NotImplementedError

    def divide_by(self, value):
        self.write(self.read() / value)

    def multiply_by(self, value):
        self.write(self.read() * value)

    def __lt__(self, scalar_value):
        return self.read() < scalar_value

    def __le__(self, scalar_value):
        return self.read() <= scalar_value

    def __gt__(self, scalar_value):
        return self.read() > scalar_value

    def __ge__(self, scalar_value):
        return self.read() >= scalar_value


class MutableScalar(MutableScalarInterface):
    """This class handles the reading/writing of a mutable scalar."""

    def __init__(self, name, scalar_value):
        super().__init__(name)
        self.var = scalar_value

    def read(self):
        """Returns a scalar numpy build-in value"""
        return self.var

    def write(self, scalar_value):
        """scalar_value: python built-in type: int, float"""
        self.var = scalar_value


class MutableAttribute(MutableScalarInterface):
    """This class handles the reading/writing of a particular attribute of an object."""

    def __init__(self, name, object, attribute_name):
        super().__init__(name)
        self.object = object
        self.attribute_str = attribute_name

    def read(self):
        """Returns a scalar numpy build-in value"""
        return getattr(self.object, self.attribute_str)

    def write(self, scalar_value):
        """scalar_value: python built-in type: int, float"""
        setattr(self.object, self.attribute_str, scalar_value)


class MutableDictValue(MutableScalarInterface):
    """This class handles the reading/writing of a particular key in a dictionary."""

    def __init__(self, name, dict, dict_key):
        super().__init__(name)
        self.dict = dict
        self.dict_key = dict_key

    def read(self):
        """Returns a scalar numpy build-in value"""
        return self.dict[self.dict_key]

    def write(self, scalar_value):
        """scalar_value: python built-in type: int, float"""
        self.dict[self.dict_key] = scalar_value


class ScalarValidationSchedule(ValidationSchedule):
    """
    Both an extension and an ending condition that modifies a MutableScalar
    if there is no improvement on a metric monitored by a monitoring extension.
    If does not improve for absolute_patience, then the training stops.

    The frequence of this extension is the same as the :class:`Monitor` monitor
    parameter.

    Check the docstring of mother class ValidationSchedule
    for more information.

    Attributes:
        mutable_scalar: MutableScalar object
            the scalar to be modified, for example the learning rate
        decay_rate: float, default=2.0
            the rate at which the learning rate is decreased. (the scalar is
            divided by this decay_rate.)
        min_value: float (default None)
            the minimal value that we tolerate for the scalar.
            Below it, we stop training.
        max_value: float (default None)
            the maximal value that we tolerate for the scalar.
            Above it, we stop training.

    See mother class ValidationSchedule for the description of the other
    parameters.
    """

    def __init__(self, monitor, metric_name, mutable_scalar,
                 patience=5, max_patience=7, decay_rate=2., max_value=None,
                 min_value=None, metric_mode='min',
                 name='Scalar validation schedule'):

        self.var = mutable_scalar
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.max_value = max_value

        def process_function():
            self.var.divide_by(self.decay_rate)

        super().__init__(monitor, metric_name, process_function,
                         patience=patience, max_patience=max_patience,
                         metric_mode=metric_mode, name=name)

    def _execute(self, trainer=None, end_epoch=False):
        log = super()._execute(trainer, end_epoch)
        if not log.stop_training:
            if self.min_value and self.var < self.min_value:
                return self.log(['too small'], stop_training=True)
            elif self.max_value and self.var > self.max_value:
                return self.log(['too big'], stop_training=True)

        return log

    def display_info(self):
        var_display = self.var.read()
        return f'{var_display:5g},'


class ScalarSchedule(Extension):
    """
    Modify a MutableScalar object at specific iterations specified by
    the user.

    Parameters
    ----------
    mutable_scalar: MutableScalar object
        the scalar to be modified, for example the learning rate
    iteration_ids: list of int
        the iterations at which the mutable_scalar will be modified.
    scalar_values: list of int
        the values at which the mutable_scalar will be set at the iteration_ids
        iterations.
    """

    def __init__(self, mutable_scalar, iteration_ids, scalar_values):
        extension_name = mutable_scalar.name + ' schedule'
        super().__init__(extension_name, 1)

        self.var = mutable_scalar
        self.iteration_ids = iteration_ids
        self.scalar_values = scalar_values

    def _execute(self, trainer=None, end_epoch=False):
        if trainer.batch not in self.iteration_ids:
            return

        value = self.scalar_values[self.iteration_ids.index(trainer.batch)]
        self.var.write(value)
        return self.log([f'New {self.var.name}: {self.var.read()}'])


class ScalarLinearRange(Extension):
    """
    Decay the learning from an initial value to an end value in `n_batches`.
    """

    def __init__(self, mutable_scalar, init_val, end_val, freq, n_batches):
        super().__init__('Learning rate linear range', freq)
        self.var = mutable_scalar
        self.var.write(init_val)
        self.n_batches = n_batches
        self.decay_rate = (end_val / init_val) ** (freq / n_batches)

    def _execute(self, trainer=None, end_epoch=False):
        if trainer.batch > self.n_batches:
            return self.log(['Learning rate too small'], stop_training=True)

        self.var.multiply_by(self.decay_rate)
        return self.log([f'New learning rate: {self.var.read()}'])


class ScalarDecay(Extension):
    """
    Decay the learning by `decay` after every `freq` iterations
    """

    def __init__(self, mutable_scalar, init_val, decay, decay_start_after, freq):
        super().__init__('Learning rate simple decay', freq)
        self.var = mutable_scalar
        self.init_val = init_val
        self.decay_rate = decay
        self.decay_start_after = decay_start_after
        self.n_attempts_to_decay = 0

    def _execute(self, trainer=None, end_epoch=False):
        self.n_attempts_to_decay += 1
        if self.n_attempts_to_decay <= self.decay_start_after:
            return

        self.var.multiply_by(self.decay_rate)
        return self.log([f'New learning rate: {self.var.read()}'])


class MaxIteration(Extension):
    """Stops training when a maximal number of iterations is reached."""

    def __init__(self, max_batchs=float('Inf'), max_epochs=float('Inf')):
        super().__init__('Max Iteration', 1)
        self.max_batchs = max_batchs
        self.max_epochs = max_epochs
        if math.isinf(max_batchs) and math.isinf(max_epochs):
            raise Exception('Either max_batchs or max_epochs should be set.')

    def _execute(self, trainer=None, end_epoch=False):
        if trainer.batch > self.max_batchs:
            return self.log(['Maximal number of batches reached'], stop_training=True)
        if trainer.batch > self.max_epochs:
            return self.log(['Maximal number of epochs reached'], stop_training=True)
        return


class MaxTime(Extension):
    """Stops training when a certain amount of training time is reached"""

    def __init__(self, max_time=3600 * 48):
        super().__init__('Max Time', 1)
        self.max_time = max_time
        self.begin_time = time.time()

        self.state_dict_attributes = ("begin_time",)

    def _execute(self, trainer=None, end_epoch=False):
        if (time.time() - self.begin_time) > self.max_time:
            return self.log(['Time exceeded'], stop_training=True)
        return


class RegularFunction(Extension):
    def __init__(self, name, freq, function, on_end=False, on_start=False):
        super().__init__(name=name, freq=freq, on_end=on_end, on_start=on_start)
        self.function = function

    def _execute(self, trainer=None, end_epoch=False):
        lines, stopping = self.function(trainer=trainer, end_epoch=end_epoch)
        return self.log(lines=lines, stop_training=stopping)
