import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HandsDataset(Dataset):
    def __init__(self, train_dir, size):
        self.train_dir = train_dir
        self.files = sorted(os.listdir(self.train_dir))
        self.image_paths = []
        self.label_paths = []
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                     # normalizing range (-1, 1)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.train_dir, self.files[idx])
        image = Image.open(image_path).convert("RGB")

        orig_path = image_path.replace("no_hands", "cropped")
        orig = Image.open(orig_path).convert("RGB")

        image = self.transform(image)
        orig = self.transform(orig)

        return image, orig