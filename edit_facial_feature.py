import numpy as np
import torch
import subroutines as sr
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")

from params import batch_size, latent_dim, image_size, learning_rate

epoch_num = 18
smiling_idx = 31
alpha_list = np.linspace(-0.5, 0.5, 10)
# image_save_columns = 2
# image_save_rows = len(alpha_list) + 2 // image_save_columns
# if (image_save_rows*image_save_columns) < (len(alpha_list) + 2):
#     image_save_columns += 1


smiling_vec = torch.zeros(1, latent_dim).to(device)
not_smiling_vect = torch.zeros(1, latent_dim).to(device)

train_dataset, train_loader, test_dataset, test_loader = sr.load_data(
    batch_size, image_size=image_size, small_data=False
)
dimX_input = train_dataset[0][0].shape[1]
# load trained model
model = sr.betaVAE(dimX_input, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load(
    f"./checkpoint/vae_checkpoint_{epoch_num}.pth"
)  # load previous checkpoint, data must be present
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["opt_dict"])


count_smiling = 0
count_not_smiling = 0
model.eval()
with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(train_loader, 1):
        data = data.to(device)
        mu, logvar = model.encode(data)
        # z = model.reparameterize(mu, logvar)
        for batch in range(batch_size):
            if labels[batch][smiling_idx] == 1:
                smiling_vec += mu[batch]
                count_smiling += 1
            else:
                not_smiling_vect += mu[batch]
                count_not_smiling += 1
    smiling_vec /= count_smiling
    not_smiling_vect /= count_not_smiling


print(f"average smiling vector: {smiling_vec}")
print(f"average not smiling vector: {not_smiling_vect}")
smiling_dir = smiling_vec - not_smiling_vect
print(f"smiling direction: {smiling_dir}")

# changing feature value
plt.figure(figsize=(20, 20))
plt.axis("off")
num_samples = 10
with torch.no_grad():
    for i in range(num_samples):
        random_idx = np.random.randint(0, len(train_dataset))
        data = train_dataset[random_idx][0].unsqueeze(0).to(device)
        num_images = len(alpha_list) + 2  # original, reconstructed, and modified alpha
        #
        plt.subplot(num_samples, num_images, num_images * i + 1)
        plt.imshow(data[0].permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.title("Original")
        recon_batch, _, _ = model(data)
        plt.subplot(num_samples, num_images, num_images * i + 2)
        plt.imshow(recon_batch[0].permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.title("Reconstructed")
        for idx, alpha in enumerate(alpha_list):
            mu, logvar = model.encode(data)
            shifted_mu = mu + alpha * smiling_dir
            shifted_z = model.reparameterize(shifted_mu, logvar)
            shifted_recon_batch = model.decode(shifted_z)
            plt.subplot(num_samples, num_images, num_images * i + idx + 3)
            plt.imshow(shifted_recon_batch[0].permute(1, 2, 0).cpu())
            plt.axis("off")
            plt.title(f"alpha={alpha:.2f}", fontsize=10)

    plt.savefig(f"./images/smiling_face.png")  # ,dpi=600)
    plt.close()
