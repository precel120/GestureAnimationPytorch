import torch
from torch.utils.data import DataLoader
from dataset import HandsDataset
from train import Train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE = 256
batch_size = 32
num_epochs = 100
learning_rate = 2e-4
lambda_l1 = 50
display_interval = 10
train_dir = "./no_hands/train/like"

train_dataset = HandsDataset(train_dir, size=SIZE)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)

train = Train(device, train_loader, learning_rate, num_epochs, lambda_l1)
# train_g_losses, train_d_losses = train.train_pix2pix(display_interval)

train.generate_image('./replace.jpg')