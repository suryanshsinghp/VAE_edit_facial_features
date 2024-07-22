import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def reg(msg, verbose=False):
    if verbose:
        print(msg)


def ShapeConv(dimX, kernel_size, stride, padding):
    dimX = (dimX - kernel_size + 2 * padding) / stride + 1
    return int(np.floor(dimX))


def load_data(batch_size, image_size=128, small_data=False):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CelebA(
        root="./data", split="train", download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    test_dataset = datasets.CelebA(
        root="./data", split="test", download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # 50000 images subset- for fast testing-uncomment below
    if small_data:
        train_dataset = torch.utils.data.Subset(train_dataset, range(50000))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        test_dataset = torch.utils.data.Subset(test_dataset, range(5000))
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    return train_dataset, train_loader, test_dataset, test_loader


class betaVAE(torch.nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(betaVAE, self).__init__()
        # encoder layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
        )  # 1 input channel, 32 output channels, 3x3 kernel, stride
        self.dimX = ShapeConv(input_dim, 3, 2, 1)
        self.batch_norm1 = torch.nn.BatchNorm2d(
            32
        )  # need multiple instances of batch norms
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        )  # 32 input channel, 64 output channels, 3x3 kernel, stride
        self.dimX = ShapeConv(self.dimX, 3, 2, 1)
        self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )  # 32 input channel, 64 output channels, 3x3 kernel, stride
        self.dimX = ShapeConv(self.dimX, 3, 2, 1)
        self.batch_norm3 = torch.nn.BatchNorm2d(128)
        self.conv_extra = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
        )  # 32 input channel, 64 output channels, 3x3 kernel, stride
        self.dimX = ShapeConv(self.dimX, 3, 2, 1)
        self.batch_norm_extra = torch.nn.BatchNorm2d(256)
        self.flatten = torch.nn.Flatten()
        self.flatten_dim = (
            256 * self.dimX * self.dimX
        )  # size of flattened image, current image is square
        self.fc1 = torch.nn.Linear(self.flatten_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3_1 = torch.nn.Linear(256, latent_dim)  # Mean of latent space
        self.fc3_2 = torch.nn.Linear(256, latent_dim)  # Log variance of latent space

        self.activation = torch.nn.ReLU()
        if loss_function == "mse":
            self.activation_final = torch.nn.Sigmoid()
        else:
            self.activation_final = torch.nn.Sigmoid()

        # Decoder layers
        self.fc4 = torch.nn.Linear(latent_dim, 256)
        self.fc5 = torch.nn.Linear(256, 512)
        self.fc6 = torch.nn.Linear(512, self.flatten_dim)
        self.deconv_extra = torch.nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.batch_norm_extra2 = torch.nn.BatchNorm2d(128)
        self.deconv3 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.batch_norm4 = torch.nn.BatchNorm2d(64)
        self.deconv2 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.batch_norm5 = torch.nn.BatchNorm2d(32)
        self.deconv1 = torch.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.batch_norm6 = torch.nn.BatchNorm2d(3)

    def encode(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activation(x)
        x = self.conv_extra(x)
        x = self.batch_norm_extra(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return self.fc3_1(x), self.fc3_2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc4(z)
        z = self.activation(z)
        z = self.fc5(z)
        z = self.activation(z)
        z = self.fc6(z)
        z = self.activation(z)
        z = z.view(-1, 256, self.dimX, self.dimX)
        z = self.deconv_extra(z)
        z = self.batch_norm_extra2(z)
        z = self.activation(z)
        z = self.deconv3(z)
        z = self.batch_norm4(z)
        z = self.activation(z)
        z = self.deconv2(z)
        z = self.batch_norm5(z)
        z = self.activation(z)
        z = self.deconv1(z)
        z = self.batch_norm6(z)
        z = self.activation_final(z)

        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar  # mu and logvar are needed for loss function


def loss_function(recon_x, x, mu, logvar, beta=0.01):

    if loss_function == "mse":
        loss1 = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")
    else:
        loss1 = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction="mean")
    # KL divergence loss
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return loss1 + KLD * beta


def test_loss(test_loader, model, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader, 1):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    return test_loss / i


def IsConverged(
    loss_metric,
    loss_history_train,
    loss_history_test,
    epoch,
    converge_diff_percent,
    model,
    optimizer,
):
    """
    stop training and save if change in loss is less than converge_diff_percent
    could change this to consider previous n epochs
    """
    if len(loss_history_train) > 1:
        if loss_metric == "test":
            loss_prcnt_chng = (
                loss_history_test[-2] - loss_history_test[-1]
            ) / loss_history_test[-2]
        else:  # default to train
            loss_prcnt_chng = (
                loss_history_train[-2] - loss_history_train[-1]
            ) / loss_history_train[-2]
        if loss_prcnt_chng < converge_diff_percent / 100:
            print(f"Converged at epoch {epoch}")
            torch.save(
                {"state_dict": model.state_dict(), "opt_dict": optimizer.state_dict()},
                f"./checkpoint/vae_checkpoint_{epoch}.pth",
            )
            return True
    return False


def save_plot(epoch, num_img, idx_list, model, dataset, device, prefixStr=""):
    """
    First time define list of random images to save
    After that, use the same list to save images so we can compare reconstructions at different epochs
    """
    if idx_list is None:
        # idx_list = np.random.randint(0,len(dataset),5)
        idx_list = np.random.choice(len(dataset), size=num_img, replace=False)
    with torch.no_grad():
        for i, idx in enumerate(idx_list):
            # idx = np.random.randint(0,len(test_dataset))
            data = (
                dataset[idx][0].unsqueeze(0).to(device)
            )  # unsqueeze to add fake batch dimension
            recon_batch, _, _ = model(data)
            plt.subplot(1, 2, 1)
            plt.imshow(
                data[0].permute(1, 2, 0).cpu()
            )  # torch permute to move channel to last dimension
            plt.subplot(1, 2, 2)
            plt.imshow(recon_batch[0].permute(1, 2, 0).cpu())
            plt.savefig(
                "./images/" + prefixStr + "_epoch_" + str(epoch) + "_" + str(i) + ".png"
            )
            plt.close()
    return idx_list
