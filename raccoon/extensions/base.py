

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
        on_start (bool, default False):
            Executes the extension at the start of training.
        on_end (bool, default False):
            Executes the extension at the end of training or when training is interrupted.
        state_dict_attributes (iterable of strings):
            Specifies which attributes of the extension should be saved in the state_dict of the
            extension. This is useful when you want to use checkpointing.

    Notes:
        - if you want to save the state of the extension during training, you should a) add a
            Checkpoint extension to the Trainer object, and either b.1) specify
            state_dict_attributes to be saved or b.2) overwrite the state_dict and load_state_dict
            methods.
    """

    def __init__(self, name, freq, on_end=False, on_start=False):
        self.name = name
        self.freq = freq
        self.on_end = on_end
        self.on_start = on_start

        self.state_dict_attributes = None

    def execute(self, trainer=None, end_epoch=False):
        """This method is called at every minibatch of the training process.

        Args:
            trainer (Trainer class): The trainer object to which the extension is registered.
                You can access its attributes, such as the current batch or epoch IDs.
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
            freq_cond = not (trainer.batch % self.freq)

        if not freq_cond:
            return self.log()

        log = self._execute(trainer, end_epoch=end_epoch)
        return log if log else self.log()

    def _execute(self, trainer=None, end_epoch=False):
        """Performs the operations of the extension. Should be overriden in the child class.

        Returns:
            either:
                - a Log object created with self.log method ,
                - or None if nothing has to be logged.
        """
        raise NotImplementedError

    def start(self, trainer=None):
        """Re-implement this method if you want a custom behavior at the beginning of training."""
        return self._execute(trainer=trainer)

    def finish(self, trainer=None):
        """Re-implement this method if you want a custom behavior at the end of training."""
        return self._execute(trainer=trainer)

    def log(self, lines=None, stop_training=False):
        """Creates a log object to potentially display information in the terminal."""
        return Log(extension=self, lines=lines, stop_training=stop_training)

    def state_dict(self):
        all_attributes = vars(self)
        return {k: all_attributes[k] for k in self.state_dict_attributes}

    def load_state_dict(self, state_dict):
        vars(self).update(state_dict)


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
