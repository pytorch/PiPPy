import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


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

        train_dl = torch.utils.data.DataLoader(
             DemoDataset(args.batch_size), batch_size=args.batch_size
         )
        test_dl = torch.utils.data.DataLoader(
            DemoDataset(args.batch_size), batch_size=args.batch_size
        )

        return train_dl, test_dl

    def get_model(self):
        class DemoModel(torch.nn.Module):
            def __init__(self):
                super(DemoModel, self).__init__()
                self.l1 = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.l1(x)

        return DemoModel()


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
        train_ds = datasets.MNIST(
            ".", download=True, train=True, transform=transform
        )
        test_ds = datasets.MNIST(".", download=True, transform=transform)
        train_data_loader = torch.utils.data.DataLoader(
             train_ds, **train_kwargs
         )
        test_data_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)
        return train_data_loader, test_data_loader


    def get_model(self):
        class MnistModel(torch.nn.Module):
            def __init__(self):
                super(MnistModel, self).__init__()
                self.fc1 = torch.nn.Linear(28 * 28, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, 10)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.relu(x)
                x = self.fc3(x)
                return x

        return MnistModel()
