from pathlib import Path
import pickle

from raccoon.extensions.base import Extension


class Saver(Extension):
    """Extension to save objects.

    Attributes:
        folder_path (string): Path where to store the object.
        file_name (string): Name under which the ibject is saved.
        fun_save (function): Custom function to save the object. It takes a file_path
            as single argument, which is where the object should be saved.
        fun_restore (function): Custom function to restore the object. It takes a file_path
            as single argument, which is where the object should be restored from.
        restore_path (string, default None): Path of a the saved object that should be restored
            at the beginning of training.
    """

    def __init__(self, name, freq, folder_path, file_name,
                 on_end=True, on_start=False,
                 fun_save=None, fun_restore=None, restore_path='', **kwargs):
        super().__init__(name,
                         freq,
                         on_end=on_end,
                         on_start=on_start)
        self.folder_path = Path(folder_path)
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name
        self.file_path = self.folder_path / file_name
        self.fun_save = fun_save
        self.fun_restore = fun_restore
        self.restore_path = restore_path

        if restore_path:
            self.on_start = True

    def start(self):
        if not self.restore_path:
            return super().start()

        self.restore(self.restore_path)
        return self.log([f'Object loaded from disk path: {self.restore_path}'])

    def _execute(self, batch_id, epoch_id, end_epoch=False):
        return self.log(self.save())

    def save(self, file_path=None):
        """
        Should return a list of strings, each string representing a line to
        be printed when the extension is executed.
        """
        if not file_path:
            file_path = self.file_path

        res = self._save(file_path)
        return res if res else [self.name + f' dumped into {self.folder_path.resolve()}']

    def _save(self, file_path):
        if self.fun_save:
            return self.fun_save(file_path)
        raise NotImplementedError

    def restore(self, file_path=None):
        if not file_path:
            file_path = self.file_path

        return self._restore(file_path)

    def _restore(self, file_path):
        if self.fun_restore:
            return self.fun_restore(file_path)
        raise NotImplementedError


class MetricSaver(Saver):
    """Saves the history of a ValidationMonitor extension"""

    def __init__(self, metric_monitor, folder_path, freq=None, name=None):

        if name is None:
            name = metric_monitor.name
        if freq is None:
            freq = metric_monitor.freq
        super().__init__('Metric saver ' + name, freq, folder_path, 'metric_saver_' + name)
        self.metric_monitor = metric_monitor

    def _save(self, file_path):

        with open(file_path, 'wb') as file_handle:
            d = {'iterations': self.metric_monitor.iterations,
                 'history': self.metric_monitor.history}
            pickle.dump(d, file_handle)
            file_handle.close()

        return [self.name + f' Metric histories dumped into {self.folder_path.resolve()}']


class ModelSaver(Saver):
    """Saves the model at specified frequency"""

    def __init__(self, freq, folder_path, fun_save):
        super().__init__('Model saver', freq, folder_path, 'model_at_iter', fun_save=fun_save)

    def _execute(self, batch_id, epoch_id, end_epoch=False):
        return self.log(self.save(file_path=self.folder_path / f"{self.file_name}_{batch_id}.net"))


class MonitorObjectSaver(Saver):
    """
    Saves an object based on a monitor extension

    Attributes:
        monitor_extension: MetricMonitor object
            the monitoring extension containing the metric that this extension
            is monitoring
        metric_name: string
            the name of the metric being monitored
        object_name: string
            the name of the object being saved (to be displayed)
        restore_path: string
            if at beginning of training, you want to restore the object from
            an existing dumped one.
        dont_save_for_first_n_it (int, default None): If int, the object is not saved at
            the beginning of training. This is useful to avoid too many copies when improvement
            is fast at the beginning of training.
    """

    def __init__(self, monitor_extension, metric_name, folder_path,
                 fun_save, object_name, fun_restore=None,
                 restore_at_the_end=True, file_name='best.obj',
                 on_end=True, on_start=False,
                 metric_mode='min', dont_save_for_first_n_it=None, freq=False,
                 restore_path=""):
        if not freq:
            freq = monitor_extension.freq
        super().__init__(f'Best {object_name} Saver', freq, folder_path,
                         file_name, on_end, on_start, fun_save=fun_save,
                         fun_restore=fun_restore, restore_path=restore_path)

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

        self.restore_at_the_end = restore_at_the_end
        self.best_value = self.m * float('Inf')
        self.dont_dump_for_first_n_it = dont_save_for_first_n_it
        self.n_times_checked = 0

    def _execute(self, batch_id, epoch_id, end_epoch=False):

        # Check if dont_save_for_first_n_it has passed
        self.n_times_checked += 1

        # Check if the validation monitor has indeed recorded values
        if not self.monitor_extension.history:
            raise Exception('MonitorObjectSavershould be placed after the'
                            'validation monitor in the list of extensions'
                            'provided to the Trainer object.')

        current_value = self.monitor_extension.history[self.metric_name][-1]
        if self.m * current_value < self.m * self.best_value:
            self.best_value = current_value
            # Check if dont_save_for_first_n_it has passed
            if not ((self.dont_dump_for_first_n_it is None) or
                    (self.n_times_checked >= self.dont_dump_for_first_n_it)):
                return

        # If it has never been dumped, we dump it.
        if (self.dont_dump_for_first_n_it is not None and
            self.dont_dump_for_first_n_it == self.n_times_checked):
            return

        return super()._execute(batch_id, epoch_id, end_epoch)

    def finish(self, batch_id, epoch_id):
        log = self.log()

        # The object has not yet been saved on the disk, so we save it
        if self.dont_dump_for_first_n_it is not None:
            if self.n_times_checked < self.dont_dump_for_first_n_it:
                self.save()
                log.lines.append(f'Best {self.object_name} '
                                 f'dumped into {self.folder_path.resolve()}')

        if self.restore_at_the_end:
            self.restore()
            log.lines.append(f'... best {self.object_name} re-loaded')

        return log
