# Add at the top of the file
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Define a simple VGG perceptual module
class VGGPerceptual(nn.Module):
    def __init__(self, layers=(3, 8, 15), device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:l+1]))
            prev = l+1
        for p in self.parameters():
            p.requires_grad = False
        self.device = device
        self.to(device)

    def forward(self, x):
        outs = []
        cur = x
        for s in self.slices:
            cur = s(cur)
            outs.append(cur)
        return outs

def vgg_loss(fake, target, vgg_model):
    f_fake = vgg_model(fake)
    f_real = vgg_model(target)
    loss = 0
    for a, b in zip(f_fake, f_real):
        loss += F.l1_loss(a, b)
    return loss
