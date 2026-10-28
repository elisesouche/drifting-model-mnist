import torchvision
import torch

augment_mnist = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomAffine(
            degrees=10, translate=(0.15, 0.15), scale=(0.7, 1.2)
        ),
        torchvision.transforms.ToTensor(),
    ]
)


def get_dataset(train: bool):
    return torchvision.datasets.MNIST(
        "MNIST", train, download=True, transform=augment_mnist
    )


def get_loader(train: bool, batch: int):
    return torch.utils.data.DataLoader(
        get_dataset(train), batch_size=batch, shuffle=True
    )
