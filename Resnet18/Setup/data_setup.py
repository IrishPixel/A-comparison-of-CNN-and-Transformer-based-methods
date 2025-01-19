import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str = None,
        test_dir: str = None,
        transform: transforms.Compose = None,
        batch_size: int = 1,
        num_workers: int = NUM_WORKERS
        ):
    train_dataloader = None
    class_names = None

    if train_dir:
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        train_dataloader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
                )
        class_names = train_data.classes

    test_dataloader = None
    if test_dir:
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        test_dataloader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
                )
        if not class_names:
            class_names = test_data.classes
    return train_dataloader, test_dataloader, class_names
