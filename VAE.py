# python -u VAE.py > out.dat  to run
# %% load lib
import time
import os
import numpy as np
import torch
from torchinfo import summary
import subroutines as sr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")

# %% define inputs
from params import load_from_checkpoint, epoch_start, num_epochs, loss_function, batch_size, image_size, save_freq, loss_metric, learning_rate, converge_diff_percent, latent_dim, beta, num_img_save, delete_prev_data


if delete_prev_data:
    if os.path.exists("./images"):
        os.system("rm -r ./images")
    if os.path.exists("./checkpoint"):
        os.system("rm -r ./checkpoint")

if not os.path.exists("./images"):
    os.makedirs("./images")
if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")


print("selected batch size", batch_size)

train_dataset, train_loader, test_dataset, test_loader = sr.load_data(
    batch_size, image_size=image_size, small_data=False
)
# random_indices = np.random.choice(num_data_points, size=x, replace=False)

# size of image
print("Number of training examples:", len(train_loader.dataset))

print("size of images: ", train_dataset[0][0].shape)
print(f"shape of input image is : {train_dataset[0][0].shape}")
dimX_input = train_dataset[0][0].shape[1]
dimY_input = train_dataset[0][0].shape[2]

# %% main code

loss_history_train = []
loss_history_test = []
image_list_test = None
image_list_train = None
model = sr.betaVAE(dimX_input, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
summary(model, input_size=(batch_size, 3, dimX_input, dimY_input))

if load_from_checkpoint:
    checkpoint = torch.load(
        f"./checkpoint/vae_checkpoint_{epoch_start-1}.pth"
    )  # load previous checkpoint, data must be present
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["opt_dict"])
    model.train()  # make sure model is in training mode


start_time = time.time()
for epoch in range(epoch_start, epoch_start + num_epochs):
    model.train()  # test loss will put model in eval mode
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader, 1):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = sr.loss_function(recon_batch, data, mu, logvar, beta=beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} [Batch: {batch_idx}]\tLoss: {loss.item()}")

    if epoch % save_freq == 0:
        print(f"Saving model checkpoint at epoch: {epoch}")
        torch.save(
            {"state_dict": model.state_dict(), "opt_dict": optimizer.state_dict()},
            f"./checkpoint/vae_checkpoint_{epoch}.pth",
        )

    train_loss /= batch_idx

    loss_history_train.append(train_loss)
    loss_history_test.append(sr.test_loss(test_loader, model, device))

    print(f"Epoch: {epoch} :::::::::::::: Train loss: {loss_history_train[-1]}")
    print(f"Epoch: {epoch} :::::::::::::: Test loss: {loss_history_test[-1]}")
    image_list_test = sr.save_plot(
        epoch, num_img_save, image_list_test, model, test_dataset, device, "test"
    )  # check on test image data
    image_list_train = sr.save_plot(
        epoch, num_img_save, image_list_train, model, train_dataset, device, "train"
    )  # check on train image data

    if sr.IsConverged(
        loss_metric,
        loss_history_train,
        loss_history_test,
        epoch,
        converge_diff_percent,
        model,
        optimizer,
    ):
        break

print(f"Training time: {(time.time()-start_time)/60} minutes")
