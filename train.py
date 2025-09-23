import time
import torch
import torch.optim as optim
import cv2
import os
from tqdm.auto import tqdm
import glob
from models import Generator, Discriminator, Pix2Pix
import numpy as np
from perceptual import vgg_loss, VGGPerceptual

class Train():
    def __init__(self, device, train_loader, learning_rate, epochs, lambda_l1, lambda_vgg):
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.pix2pix = Pix2Pix(self.generator, self.discriminator).to(device)
        self.epochs = epochs
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.train_loader = train_loader
        self.vgg_model = VGGPerceptual(device=device)
        self.load_checkpoint()

    def train_pix2pix(self, display_interval):
        start_time = time.time()
        train_g_losses, train_d_losses = [], []

        for epoch in range(self.epochs):

            # training
            self.pix2pix.train()
            train_g_loss, train_d_loss = 0.0, 0.0

            for _, (segmented_images, real_images) in enumerate(tqdm(self.train_loader, desc=f"Training {epoch+1}/{self.epochs}")):
                segmented_images = segmented_images.to(self.device)
                real_images = real_images.to(self.device)

                # train model discriminator
                self.d_optimizer.zero_grad()
                fake_images = self.pix2pix.generator(segmented_images)
                real_output = self.pix2pix.discriminator(segmented_images, real_images)
                fake_output = self.pix2pix.discriminator(segmented_images, fake_images)
                d_loss = self.pix2pix.discriminator_loss(real_output, fake_output)
                d_loss.backward()
                self.d_optimizer.step()

                # train model generator
                self.g_optimizer.zero_grad()
                fake_images = self.pix2pix.generator(segmented_images)
                fake_output = self.pix2pix.discriminator(segmented_images, fake_images)
                g_loss = self.pix2pix.generator_loss(fake_output, fake_images, real_images, self.lambda_l1)

                loss_vgg = vgg_loss(fake_images, real_images, self.vgg_model)
                g_loss_total = g_loss + self.lambda_vgg * loss_vgg

                g_loss_total.backward()
                self.g_optimizer.step()

                train_d_loss += d_loss.item()
                train_g_loss += g_loss_total.item()

            # average train loss per epoch
            train_d_losses.append(train_d_loss / len(self.train_loader))
            train_g_losses.append(train_g_loss / len(self.train_loader))

            # validating
            self.pix2pix.eval()
            print(f"Epoch {epoch + 1}/{self.epochs} | G Train Loss: {train_g_losses[-1]:.4f} | D Train Loss: {train_d_losses[-1]:.4f}")

            # display sample results after each interval
            if epoch == 0 or (epoch + 1) % display_interval == 0 or (epoch + 1) == self.epochs:
                self.save_samples(self.pix2pix, epoch + 1)
            
            if (epoch + 1) % display_interval == 0 or (epoch + 1) == self.epochs:
                self.save_checkpoint(epoch + 1, train_g_losses, train_d_losses)


        print(f"Training Completed in {time.time() - start_time:.2f} seconds.")
        return train_g_losses, train_d_losses
    
    def save_checkpoint(self, epoch, train_g_losses, train_d_losses):
        os.makedirs("./checkpoints", exist_ok=True)
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'train_g_losses': train_g_losses,
            'train_d_losses': train_d_losses,
        }
        torch.save(checkpoint, f"./checkpoints/e{epoch}_lambda{self.lambda_l1}.pth")
        print(f"Checkpoint saved for epoch {epoch}.")
    
    def load_checkpoint(self):
        if os.listdir('./checkpoints').__len__() > 0:
            checkpoint_path = glob.glob('./checkpoints/*')
            latest_file = max(checkpoint_path, key=os.path.getmtime)        
            checkpoint = torch.load(latest_file)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.train_g_losses = checkpoint['train_g_losses']
            self.train_d_losses = checkpoint['train_d_losses']
            print("Checkpoint loaded.")
        else:
            print("No checkpoint found. Starting from scratch.")

    def save_samples(self, model, epoch, num_samples=10):
        self.pix2pix.eval()
        with torch.no_grad():
            segmented_images, _ = next(iter(self.train_loader))
            segmented_images = segmented_images[:num_samples].to(self.device)

            fake_images = model.generator(segmented_images)

            fake_images = (fake_images + 1) / 2  # scale to [0, 1] range
            fake_images = fake_images.cpu().numpy().transpose(0, 2, 3, 1)

            cv2.imwrite(f"./output/generated_{epoch}.jpg", cv2.cvtColor(fake_images[0], cv2.COLOR_RGB2BGR) * 255)
    
    def generate_image(self, image_path, output_path="./generated_single.jpg"):
        """
        Generate an image using the trained generator from a single input image.
        
        Args:
            image_path (str): Path to the input image (expected to be a segmented image).
            output_path (str): Path to save the generated image.
        """
        self.pix2pix.eval()
        with torch.no_grad():
            # Load and preprocess the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))  # Resize to match model input size
            image = image / 255.0  # Normalize to [0, 1]
            image = (image * 2) - 1  # Scale to [-1, 1] to match training data
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor and add batch dimension
            image = image.to(self.device)

            # Generate image
            generated_image = self.pix2pix.generator(image)

            # Post-process the generated image
            generated_image = (generated_image + 1) / 2  # Scale back to [0, 1]
            generated_image = generated_image.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to numpy
            generated_image = (generated_image[0] * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
            generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Save the generated image
            cv2.imwrite(output_path, generated_image)
            print(f"Generated image saved to {output_path}")