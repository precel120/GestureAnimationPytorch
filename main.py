import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import HandsDataset
from train import Train
from prepare_images import prepare_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SIZE = 256
batch_size = 16
num_epochs = 100
learning_rate = 1e-4
lambda_l1 = 20
lambda_vgg = 0.05
display_interval = 10
train_dir = "./"
GESTURES = ['like', 'three_gun']

# prepare_images(GESTURES, SIZE)

train_dataset = HandsDataset(train_dir, GESTURES, size=SIZE)
sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, sampler=sampler, persistent_workers=True)

train = Train(device, train_loader, learning_rate, num_epochs, lambda_l1, lambda_vgg)
# train_g_losses, train_d_losses = train.train_pix2pix(display_interval)

for dirpath, _, filenames in os.walk('./gesture_seq'):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        train.generate_image(file_path, output_path='./replace.jpg')