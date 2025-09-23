import random
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HandsDataset(Dataset):
    def __init__(self, base_dir, gestures, size, max_per_gesture=1000):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                     # normalizing range (-1, 1)
        ])
        self.samples = []
        for gesture in gestures:
            skel_dir = os.path.join(base_dir, f"no_hands/train/{gesture}")
            real_dir = os.path.join(base_dir, f"cropped/train/{gesture}")
            skel_files = sorted(os.listdir(skel_dir))[:max_per_gesture]
            for f in skel_files:
                skel_path = os.path.join(skel_dir, f)
                real_path = os.path.join(real_dir, f)
                if os.path.exists(real_path):
                    self.samples.append((skel_path, real_path))

        self.weights = [1.0 / len(gestures)] * len(self.samples)  # Equal weight per gesture

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skel_img = Image.open(self.samples[idx][0]).convert("RGB")
        real_img = Image.open(self.samples[idx][1]).convert("RGB")

        # Apply same seed for paired augmentation
        seed = random.randint(0, 2**32)
        random.seed(seed)
        skel = self.transform(skel_img)
        random.seed(seed)
        real = self.transform(real_img)

        # Add noise to skeleton with 30% probability
        if random.random() < 0.3:
            skel += torch.randn_like(skel) * 0.05
            skel = torch.clamp(skel, -1, 1)

        return skel, real