load_from_checkpoint = False
epoch_start = 0  # if loading from checkpoint, change to epoch from where to resume training, will read epoch-1 data
num_epochs = 20
loss_function = "mse"  # 'mse' or 'bce'
batch_size = 128
image_size = 64  # number of pixels, Square image
save_freq = 2  # how often Save model
loss_metric = "train"  # 'train' or 'test
learning_rate = 0.5e-3
converge_diff_percent = 1e-6  # if difference between two consecutive losses is less than this, consider it converged, divided by 100 later
latent_dim = 128
beta = 0.0008
num_img_save = 3
delete_prev_data = True  # delete previous images and checkpoints

print("load from previous checkpoint: ", load_from_checkpoint)
print("epoch start number: ", epoch_start)
print("Total number of epochs: ", num_epochs)
print("loss function: ", loss_function)
print("batch size: ", batch_size)
print("image size is : ", image_size)
print("saving every ", save_freq, " num of epochs")
print("track loss based on train or test?: ", loss_metric)
print("learning rate: ", learning_rate)
print(
    "stop learning when diff between two successive loss is less than: ",
    converge_diff_percent,
)
print("latent dimension: ", latent_dim)
print("beta: ", beta)
print("saving ", num_img_save, " num of train and test image at every epoch")
if delete_prev_data:
    print("deleting previous data")


"""
Notes:
beta 10 is high, all pics look the same ***** Update: there was a bug in code, so this may not be true
use latent space more than 100
"""
