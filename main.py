import torch
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import HandsDataset
from train import Train
# from models import Generator, Discriminator
# from torchviz import make_dot

# from animation import create_animation
# from prepare_images import prepare_images

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

# generator = Generator().to(device).eval()
# discriminator = Discriminator().to(device).eval()

# # Create dummy input tensors
# dummy_input = torch.randn(1, 3, 256, 256).to(device)
# dummy_real = torch.randn(1, 3, 256, 256).to(device)

# # Forward pass through generator
# fake_output = generator(dummy_input)
# dot_gen = make_dot(fake_output, params=dict(generator.named_parameters()))
# dot_gen.format = 'png'
# dot_gen.render('generator', cleanup=True)

# # Forward pass through discriminator
# disc_output = discriminator(dummy_input, dummy_real)
# dot_disc = make_dot(disc_output, params=dict(discriminator.named_parameters()))
# dot_disc.format = 'png'
# dot_disc.render('discriminator', cleanup=True)

train = Train(device, train_loader, learning_rate, num_epochs, lambda_l1, lambda_vgg)
# train_g_losses, train_d_losses = train.train_pix2pix(display_interval)

# for dirpath, _, filenames in os.walk('./gesture_seq'):
#     for filename in filenames:
#         file_path = os.path.join(dirpath, filename)
#         train.generate_image(file_path, output_path=file_path.replace('gesture_seq', 'frames'))

# create_animation('./frames', './animation.mp4')