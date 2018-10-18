

class Extension(object):
    """
    Extensions represent objects that perform various operations during the training process.
    They can also interrupt the training process.

    Extension instances are provided to a :class:`Trainer` instance. Their method `execute` is
    called every `freq` minibatches by the `Trainer` instance.

    If you inherit from Extension, the only methods that you should override are `__init__` and
    `_execute`.

    Attributes:
        name (string):
            The name of the extension that will be displayed when the extension is executed.
        freq (int or 'epoch' or None):
            - int: The frequency at which the extension is called.
            - None: The extension is never called (or maybe at the beginning or the end of
                training).
            - 'epoch', The extension is called at the end of each epoch.
        apply_start (bool, default False):
            Applies the extension at the start of training.
        apply_end (bool, default False):
            Applies the extension at the end of training or when training is interrupted.
    """

    def __init__(self, name, freq, apply_end=False, apply_start=False):
        self.name = name
        self.freq = freq
        self.apply_end = apply_end
        self.apply_start = apply_start

    def execute(self, batch_id, epoch_id, end_epoch=False):
        """This method is called at every minibatch of the training process.

        Args:
            batch_id (int): The batch id at which the extension is called.
            epoch_id (int): The epoch id at which the extension is called.
            end_epoch (bool): Indicates if it's the last minibatch of an epoch.

        Returns:
            Log instance: a `Log` object that contains information to print to the users
                and whether or not the training process should be stopped.
        """
        if not self.freq:
            freq_cond = False

        elif self.freq == 'epoch':
            freq_cond = end_epoch

        else:
            freq_cond = not (batch_id % self.freq)

        if not freq_cond:
            return self.log()

        log = self._execute(batch_id, epoch_id, end_epoch=end_epoch)
        return log if log else self.log()

    def _execute(self, batch_id, epoch_id, end_epoch=False):
        """Performs the operations of the extension. Should be overriden in the child class.

        Returns:
            either:
                - a Log object created with self.log method ,
                - or None if nothing has to be logged.
        """
        raise NotImplementedError

    def start(self):
        """Re-implement this method if you want a custom behavior at the beginning of training."""
        return self._execute(0, 0)

    def finish(self, batch_id, epoch_id):
        """Re-implement this method if you want a custom behavior at the end of training."""
        return self._execute(batch_id, epoch_id)

    def log(self, lines=None, stop_training=False):
        """Creates a log object to potentially display information in the terminal."""
        return Log(extension=self, lines=lines, stop_training=stop_training)


class Log:
    """A Log instance is the type of object returned to the `Trainer` instance by the `execute`
    methods of extensions.

    It contains information to print the status of the extension or to stop training.

    Attributes:
        extension (Extension): The extension that has created the Log object.
        lines (list of strings, default None): the lines of the Log that will be printed by the
            `Trainer` object. If None, no information will be printed.
        stop_training (bool): Whether or not the training should be stopped.
    """
    def __init__(self, extension, lines=None, stop_training=False):
        self.extension = extension
        self.lines = [] if lines is None else lines
        self.stop_training = stop_training
