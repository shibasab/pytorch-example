import torch
import torchvision
import torchvision.transforms as transforms


def load_cifer10(batch_size, num_workers=6, data_dir='~/data', download=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    return train_dataset, test_dataset, train_loader, test_loader


if __name__ == '__main__':
    train_dataset, train_loader, test_dataset, test_loader = load_cifer10(
        batch_size=4, num_workers=6, data_dir='~/data', download=True)

    print(train_dataset, train_loader, test_dataset, test_loader)
