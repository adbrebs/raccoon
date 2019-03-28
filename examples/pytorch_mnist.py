"""Highly inspired from https://github.com/pytorch/examples/tree/master/mnist."""

import os
import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from raccoon import (
    TrainMonitor, ValidationMonitor, MaxIteration, MutableDictValue, ScalarValidationSchedule,
    Checkpoint, MonitorObjectSaver)
from raccoon.trainer import Trainer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_data_generator(train_set, batch_size):
    data_folder = './data'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_folder, train=train_set, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)

    def gen():
        for data, target in loader:
            yield {"data": data, "target": target}
    return gen


if __name__ == '__main__':

    ###################
    # Data generators #
    ###################
    train_data_gen = create_data_generator(True, 64)
    valid_data_gen = create_data_generator(False, 1000)

    ####################
    # Model definition #
    ####################
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def train(input_dictionary):
        data, target = input_dictionary["data"], input_dictionary["target"]
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        return {"nll": loss.item()}

    def valid(input_dictionary):
        data, target = input_dictionary["data"], input_dictionary["target"]
        model.eval()
        with torch.no_grad():
            output = model(data)
            test_loss = F.nll_loss(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item() / len(target)

        return {"nll": test_loss, "acc": correct}

    ###########################
    # Monitoring with Raccoon #
    ###########################
    dump_folder = "./dump"
    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)

    train_monitor = TrainMonitor(
        fun_batch_metrics=train,
        metric_names=["nll"],
        freq=1)

    valid_monitor = ValidationMonitor(
        name="Validation",
        fun_batch_metrics=valid,
        metric_names=["nll", "acc"],
        freq=1000,
        data_generator=valid_data_gen,
        on_start=True)

    max_epoch = MaxIteration(
        max_epochs=10)

    mutable_lr = MutableDictValue("lr", optimizer.defaults, "lr")
    lr_schedule = ScalarValidationSchedule(
        valid_monitor, "acc",
        mutable_scalar=mutable_lr,
        patience=2,
        max_patience=3,
        decay_rate=2,
        metric_mode="max",
        name="Learning rate schedule")

    best_net_saver = MonitorObjectSaver(
        monitor_extension=valid_monitor,
        metric_name="acc",
        folder_path=dump_folder,
        fun_save=lambda path: torch.save(model.state_dict(), path),
        object_name="best_model.net",
        metric_mode="max")

    extensions = [valid_monitor, max_epoch, lr_schedule, best_net_saver]

    def fun_save_state():
        return {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr": mutable_lr.read()}

    def fun_load_state(state_dict):
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        mutable_lr.write(state_dict["lr"])


    checkpoint = Checkpoint(extensions, dump_folder, 3,
                            fun_save=fun_save_state,
                            fun_load=fun_load_state,
                            on_end=False)

    trainer = Trainer(
        train_monitor=train_monitor,
        data_generator=train_data_gen,
        extensions=extensions + [checkpoint])

    trainer.train()
