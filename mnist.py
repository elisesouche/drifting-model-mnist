import torchvision
import torch

prep_mnist = torchvision.transforms.ToTensor()


def get_augment(augment: bool):
    if augment:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomAffine(
                    degrees=10, translate=(0.15, 0.15), scale=(0.7, 1.2)
                ),
                prep_mnist,
            ]
        )
    else:
        return prep_mnist


def get_dataset(train: bool, augment: bool = True):
    return torchvision.datasets.MNIST(
        "MNIST", train, download=True, transform=get_augment(augment)
    )


def get_loader(
    train: bool,
    batch: int,
    augment: bool = False,
):
    return torch.utils.data.DataLoader(
        get_dataset(train, augment=augment),
        batch_size=batch,
        shuffle=True,
        drop_last=True,
    )
