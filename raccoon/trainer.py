from collections import defaultdict
from contextlib import contextmanager
import sys
import time

from raccoon.utils import wrap_text, print_green, print_red, str_magenta, str_yellow, pretty_time


class Trainer:
    """
    Attributes:
        train_monitor (TrainMonitor):
            The TrainMonitor responsible for training.
        data_generator (function generator):
            Function generator that returns minibatches. Its end marks the end of an epoch.
        extensions (list of Extension objects, default None):
            The list of extensions that are regularly run during training.
        after_epoch_fun (function):
            Function that is called at the end of each epoch.
        print_wrap_width (int):
            Maximal number of character per line until the string is wrapped
        batch (int): Number of minibatches processed so far.
        epoch (int): Number of epochs processed so far.

    Notes:
        - If you want to save the current state of training, be sure to include a Checkpoint
            extension in the list of extensions that you feed to the constructor.
    """

    def __init__(self, train_monitor, data_generator, extensions=None,
                 after_epoch_fun=None, print_wrap_width=80):
        if extensions is None:
            extensions = []

        self.print_wrap_width = print_wrap_width

        self.train_monitor = train_monitor
        self.extensions = [train_monitor] + extensions
        self.extensions_wo_train = extensions
        self.data_generator = data_generator
        self.batch = self.epoch = self.begin_time = self.data_processing_time = 0
        self.after_epoch_fun = after_epoch_fun

        # Stores the total time spent in each extension
        self.total_timings = defaultdict(int)
        # Stores the time spent in each extension until it is displayed. Once displayed,
        # it is reset.
        self.cur_timings = defaultdict(int)

    def train(self):
        """Training for loop"""

        if self.batch == 0:
            self.start()

        try:
            while True:

                self.epoch += 1
                epoch_iterator = self.data_generator()

                inputs = self.get_next_minibatch(epoch_iterator)

                while True:

                    next_inputs = self.get_next_minibatch(epoch_iterator)
                    end_epoch = next_inputs is None

                    self.train_minibatch(inputs)
                    self.batch += 1

                    is_finished = self.execute_extensions(end_epoch=end_epoch)

                    if is_finished:
                        raise StopIteration

                    if end_epoch:
                        if self.after_epoch_fun:
                            self.after_epoch_fun()
                        break
                    else:
                        inputs = next_inputs

        except StopIteration:
            self.finish()

        except KeyboardInterrupt:
            self.finish_with_keyboard()

    def execute_extensions(self, end_epoch=False):
        """Returns True if trainin is interrupted by an extension. False otherwise."""
        logs = [self.execute_extension(ext, end_epoch) for ext in self.extensions]

        # If no extensions are active
        if not any(l.lines for l in logs):
            return False

        spent_time = time.time() - self.begin_time

        epoch_str = str_magenta("Epoch")
        iteration_str = str_magenta("Iteration")
        spent_time_str = str_magenta("Total time")

        n_iterations = format(self.batch, ',').replace(',', ' ')
        display_str = (f'{epoch_str} {self.epoch} - '
                       f'{iteration_str} {n_iterations} - '
                       f'{spent_time_str} {pretty_time(spent_time)}')

        self.print(display_str, 0, self.print_wrap_width + 20)  # because color counts
        self.print_logs(logs)
        self.print_lines()

        return any(l.stop_training for l in logs)

    def print_logs(self, logs):
        for log in logs:
            if not log or not log.lines:
                continue
            ext = log.extension
            timing = self.cur_timings.pop(ext)  # Reset current timings
            self.print(f'{str_yellow(ext.name)} [{timing:.2f}s]:', 1)
            for line in log.lines:
                self.print(line, 2)

    def start(self):
        self.begin_time = time.time()
        self.print_lines(2)
        print_green('Training starts!')
        self.print_lines(2)

        extensions_start = [ext for ext in self.extensions if ext.on_start]
        if not any(extensions_start):
            return

        print_green('Computing initial extensions...')
        self.print_logs([self.start_extension(ext) for ext in extensions_start])
        self.print_lines()

    def finish(self):
        time_spent = time.time() - self.begin_time
        self.print_lines(2)
        print_green(f'Training finished after {pretty_time(time_spent)}')

        extensions_end = [ext for ext in self.extensions if ext.on_end]
        if any(extensions_end):
            print_green('Computing final extensions...')
            self.print_logs([self.finish_extension(ext) for ext in extensions_end])

        # Total extension time
        total_ext_time = sum(self.total_timings[ext] for ext in self.extensions_wo_train)

        profiles = [(0, 'Data processing', self.data_processing_time),
                    (0, 'Training', self.total_timings[self.train_monitor]),
                    (0, 'Extensions', total_ext_time)]
        profiles.extend([(1, ext.name, self.total_timings[ext])
                         for ext in self.extensions_wo_train])

        time_recorded = (self.data_processing_time +
                         self.total_timings[self.train_monitor] +
                         total_ext_time)

        total_time = time.time() - self.begin_time
        profiles.append((0, 'Overhead training loop', total_time - time_recorded))

        print_green(f'Profiler, Total job time: {pretty_time(total_time)}', newline=True)
        for level, name, timing in profiles:
            name = str_yellow(name)
            display_str = f'[{timing / total_time:^7.2%}] ({pretty_time(timing)}) : {name}'
            self.print(display_str, 1 + level)

    def finish_with_keyboard(self):
        print_red('Training interrupted by user.')
        try:
            self.finish()
        except KeyboardInterrupt:
            print_red("Alright! We don't even run the extensions one last time.")
            sys.exit()

    def print(self, s, indent_level, print_wrap_width=None):
        if print_wrap_width is None:
            print_wrap_width = self.print_wrap_width
        print(wrap_text(s, indent_level, width=print_wrap_width))

    def print_lines(self, times=1):
        print(*['-' * self.print_wrap_width]*times, sep="\n")

    @contextmanager
    def time_extension(self, ext):
        s = time.time()
        yield
        total = time.time() - s
        self.total_timings[ext] += total
        self.cur_timings[ext] += total

    def get_next_minibatch(self, epoch_iterator):
        t = time.time()
        next_inputs = next(epoch_iterator, None)
        self.data_processing_time += time.time() - t
        return next_inputs

    def train_minibatch(self, inputs):
        with self.time_extension(self.train_monitor):
            return self.train_monitor.train(inputs)

    def execute_extension(self, ext, end_epoch):
        with self.time_extension(ext):
            return ext.execute(self, end_epoch)

    def start_extension(self, ext):
        with self.time_extension(ext):
            return ext.start(self)

    def finish_extension(self, ext):
        with self.time_extension(ext):
            return ext.finish(self)
