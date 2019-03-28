import os.path
import pickle

from raccoon.extensions.base import Extension


class Checkpoint(Extension):
    def __init__(self, extensions, checkpoint_path, freq, fun_save=None, fun_load=None, on_end=False):
        super().__init__("Checkpoint", freq, on_end=on_end, on_start=True)

        self.checkpoint_path = checkpoint_path
        if fun_save is not None and fun_load is not None:
            raise ValueError("Both fun_save and fun_load should be specified.")
        self.fun_save = fun_save
        self.fun_load = fun_load
        self.extensions = (ext for ext in extensions
                           if ext is not self and ext.checkpoint_attributes is not None)

    def _execute(self, batch_id, epoch_id, end_epoch=False):
        d = {ext.name: ext.state_dict() for ext in self.extensions}
        if self.fun_save:
            d["extra"] = self.fun_save()
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(d, f)

        return self.log([f"Checkpoint has been saved to {self.checkpoint_path}."])

    def start(self):
        if not os.path.isfile(self.checkpoint_path):
            return

        with open(self.checkpoint_path, "r") as f:
            d = pickle.load(f)
        for ext in self.extensions:
            ext.load_state_dict(d)
        if self.fun_load:
            self.fun_load(d["extra"])

        return self.log([f"Checkpoint {self.checkpoint_path} has been loaded back."])
