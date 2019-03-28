import os
import pickle

from raccoon.extensions.base import Extension


class Checkpoint(Extension):
    """Extension to save the current state of training and of extensions at a specific iteration.

    Attributes:
        extensions (list of Extension objects):
            The specific extensions that need to be checkpointed. Note that those extensions
            should either have a state_dict_attributes or overwrite the state_dict and
            load_state_dict methods.
        checkpoint_folder (string):
            The path of the folder where the checkpoints are dumped.
        freq (check Extension): The frequency at which the checkpoint is dumped.
        load_from_path (string, optional): The path of a possible checkpoint to restore.
        fun_save (function returning a dict {string:pickable object}, optional): Allows you to
            save additional information in the chunkpoint, for example the optimizer state.
        fun_load (function taking as input the output of fun_save):
    """
    def __init__(self, extensions, checkpoint_folder, freq, load_from_path=None,
                 fun_save=None, fun_load=None, on_end=True):
        super().__init__("Checkpoint", freq, on_end=on_end, on_start=True)

        self.checkpoint_folder = checkpoint_folder
        self.load_from_path = load_from_path
        if load_from_path and not os.path.isfile(load_from_path):
            raise ValueError(f"{load_from_path} doesn't exist.")
        if (fun_save is not None) ^ (fun_load is not None):
            raise ValueError("fun_save and fun_load should both be specified or both be None.")
        self.fun_save = fun_save
        self.fun_load = fun_load
        self.extensions = (ext for ext in extensions if ext.state_dict_attributes is not None)

    def _execute(self, trainer=None, end_epoch=False):
        d = self.create_state_dict(trainer)
        file_path = os.path.join(self.checkpoint_folder, "latest.pt")
        # Make sure that there is always a backup checkpoint in case the server crashes while
        # checkpointing.
        file_path_old = os.path.join(self.checkpoint_folder, "previous.pt")
        if os.path.isfile(file_path):
            os.rename(file_path, file_path_old)
        with open(file_path, "wb") as f:
            pickle.dump(d, f)

        return self.log([f"Checkpoint has been saved to {self.checkpoint_folder}."])

    def create_state_dict(self, trainer):
        d = {"extensions": {ext.name: ext.state_dict() for ext in self.extensions}}
        if self.fun_save:
            d["extra"] = self.fun_save()
        if trainer:
            d["trainer"] = {"batch": trainer.batch, "epoch": trainer.epoch}
        return d

    def start(self, trainer=None):
        if not self.load_from_path:
            return

        with open(self.load_from_path, "rb") as f:
            d = pickle.load(f)
        for ext in self.extensions:
            ext.load_state_dict(d["extensions"][ext.name])
        if self.fun_load:
            self.fun_load(d["extra"])
        if trainer:
            vars(trainer).update(d["trainer"])

        return self.log([f"Checkpoint {self.load_from_path} has been loaded back."])

    def finish(self, trainer=None):
        d = self.create_state_dict(trainer)
        file_path = os.path.join(self.checkpoint_folder, "final.pt")
        with open(file_path, "wb") as f:
            pickle.dump(d, f)

        return self.log([f"Final checkpoint has been saved to {self.checkpoint_folder}."])
