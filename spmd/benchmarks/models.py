import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.distributed._tensor import distribute_module
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Shard


class DemoConfig(torch.nn.Module):

    def get_train_and_test_data_loaders(self, args):

        class DemoDataset(torch.utils.data.Dataset):

            def __init__(self, batch_size):
                self._batch_size = batch_size
                self._x = torch.randn((batch_size * args.steps, 8))
                self._y = torch.randn((batch_size * args.steps, 8))

            def __len__(self):
                return len(self._x)

            def __getitem__(self, idx):
                return (self._x, self._y)

        train_dl = torch.utils.data.DataLoader(DemoDataset(args.batch_size), batch_size=args.batch_size)
        test_dl = torch.utils.data.DataLoader(DemoDataset(args.batch_size), batch_size=args.batch_size)

        return train_dl, test_dl

    def get_model(self, device_mesh=None):

        class DemoModel(torch.nn.Module):

            def __init__(self, device_mesh=None):
                super(DemoModel, self).__init__()
                self.l1 = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.l1(x)

        return DemoModel(device_mesh)


class MnistConfig(torch.nn.Module):

    def get_train_and_test_data_loaders(self, args):
        # Create dataloaders
        train_kwargs = {"batch_size": args.batch_size}
        test_kwargs = {"batch_size": args.batch_size}
        cuda_kwargs = {
            "num_workers": torch.cuda.device_count(),
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(cuda_kwargs)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_ds = datasets.MNIST(".", download=True, train=True, transform=transform)
        test_ds = datasets.MNIST(".", download=True, transform=transform)
        train_data_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
        test_data_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)
        return train_data_loader, test_data_loader


    def get_model(self, device_mesh=None):

        class MnistModel(torch.nn.Module):

            def __init__(self, device_mesh=None):
                super(MnistModel, self).__init__()
                # convolution ops not supported
                self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
                # Error: native_dropout_backward not supported
                self.dropout1 = torch.nn.Dropout(0.25)
                self.dropout2 = torch.nn.Dropout(0.5)
                if device_mesh:
                    self.fc1 = distribute_module(torch.nn.Linear(9216, 128), device_mesh)
                    self.fc2 = distribute_module(torch.nn.Linear(128, 10), device_mesh)

                    def input_fn(inputs):
                        return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

                    def output_fn(outputs):
                        assert isinstance(outputs, DTensor)
                        return outputs.to_local()

                    self.fc1.register_forward_pre_hook(lambda _, inputs: input_fn(inputs))  # type: ignore

                    self.fc2.register_forward_hook(
                        lambda mod, inputs, outputs: output_fn(outputs)  # type: ignore
                    )
                else:
                    self.fc1 = torch.nn.Linear(9216, 128)
                    self.fc2 = torch.nn.Linear(128, 10)


            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return x

        return MnistModel(device_mesh)
