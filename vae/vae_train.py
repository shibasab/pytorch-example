from vae_model import VAE

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


if __name__ == "__main__":

    epochs = 5
    batch_size = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))])

    mnist_dataset = datasets.MNIST('~/mnist',
                                   train=True,
                                   download=True,
                                   transform=transform)

    train_dataloader = DataLoader(mnist_dataset,
                                  batch_size=batch_size, shuffle=True, num_workers=0)

    model = VAE(in_size=28, z_dim=10, device=device, hidden_size=200)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        print("epoch : {} / {}".format(epoch, epochs))
        i = 0

        for image, label in train_dataloader:
            print("i", i)
            optimizer.zero_grad()
            image = image.to(device)
            output, z = model(image)
            loss = model.loss_sigmoid(image) / batch_size
            loss.backward()
            optimizer.step()

            if i % 3 == 1:
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch, epochs, loss.item()))

            i += 1

    torch.save(model.state_dict(), "models/vae_model.pkl")
    print('finished!')
