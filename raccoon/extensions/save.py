from pathlib import Path

from raccoon.extensions.base import Extension


class MonitorObjectSaver(Extension):
    """
    Saves an object based on a monitor extension

    Attributes:
        folder_path (string): Path where to store the object.
        fun_save (function): Custom function to save the object. It takes a folder_path
            as single argument, which is where the object should be saved.
        fun_restore (function): Custom function to restore the object. It takes a folder_path
            as single argument, which is where the object should be restored from.
        restore_path (string, default None): Path of a the saved object that should be restored
            at the beginning of training.
        monitor_extension: MetricMonitor object
            the monitoring extension containing the metric that this extension
            is monitoring
        metric_name: string
            the name of the metric being monitored
        object_name: string
            the name of the object being saved (to be displayed)
        dont_save_for_first_n_it (int, default None): If int, the object is not saved at
            the beginning of training. This is useful to avoid too many copies when improvement
            is fast at the beginning of training.

    Notes:
        - When you feed this extension to the Trainer, the order of the extensions is important.
            It should be added after its monitor_extension.
    """

    def __init__(self, monitor_extension, metric_name, folder_path,
                 fun_save, object_name, fun_restore=None,
                 on_end=True, on_start=False,
                 metric_mode='min', dont_save_for_first_n_it=None, freq=False,
                 restore_path=""):
        if not freq:
            freq = monitor_extension.freq
        super().__init__(f'Best {object_name} Saver', freq, on_end=on_end, on_start=on_start)

        self.folder_path = Path(folder_path)
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.fun_save = fun_save
        self.fun_restore = fun_restore
        self.restore_path = restore_path

        if restore_path:
            self.on_start = True

        self.object_name = object_name
        self.monitor_extension = monitor_extension
        self.metric_name = metric_name
        self.mode_metric = metric_mode
        if metric_mode == 'max':
            self.m = -1
        elif metric_mode == 'min':
            self.m = 1
        else:
            raise ValueError

        self.best_value = self.m * float('Inf')
        self.dont_dump_for_first_n_it = dont_save_for_first_n_it
        self.n_times_checked = 0

    def _execute(self, trainer=None, end_epoch=False):

        # Check if dont_save_for_first_n_it has passed
        self.n_times_checked += 1

        # Check if the validation monitor has indeed recorded values
        if not self.monitor_extension.history:
            raise Exception('MonitorObjectSavershould be placed after the'
                            'validation monitor in the list of extensions'
                            'provided to the Trainer object.')

        if (self.dont_dump_for_first_n_it and
                self.n_times_checked < self.dont_dump_for_first_n_it):
            return

        current_value = self.monitor_extension.history[self.metric_name][-1]
        if self.m * current_value < self.m * self.best_value:
            self.best_value = current_value
        else:
            return

        self.fun_save(self.folder_path)
        return self.log([self.name + f' dumped into {str(self.folder_path)}'])

    def start(self, trainer=None):
        if not self.restore_path or not self.fun_restore:
            return super().start()

        self.fun_restore(self.restore_path)
        return self.log([f'Object loaded from disk path: {self.restore_path}'])

    def finish(self, trainer=None):
        log = self.log()

        # The object has not yet been saved on the disk, so we save it
        if self.dont_dump_for_first_n_it is not None:
            if self.n_times_checked < self.dont_dump_for_first_n_it:
                self.fun_save(self.folder_path)
                log.lines.append(f'Best {self.object_name} '
                                 f'dumped into {self.folder_path.resolve()}')

        return log
