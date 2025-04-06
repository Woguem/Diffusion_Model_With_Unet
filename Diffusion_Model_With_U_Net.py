"""
@author: Dr Yen Fred WOGUEM 
@description: This script train a Diffusion Model to generate images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

start_time = datetime.now()  # Start timer

# ==============================================
# Hyperparameters
# ==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
image_size = 28
num_epochs = 3
timesteps = 1000  # Number of diffusion steps

# Data processing
# ==============================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Images in [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==============================================
# Defining the diffusion process
# ==============================================
class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        
        # Noise schedule definition (beta)
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        
        # Calculate useful variables
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x_0, t):
        """Adds noise to image x_0 at step t"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t])[:, None, None, None]
        
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def sample_timesteps(self, n):
        """Samples random timesteps"""
        return torch.randint(low=1, high=self.timesteps, size=(n,))

diffusion = Diffusion(timesteps)

# ==============================================
# Improved Denoising network architecture (U-Net)
# ==============================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim))
    
    def forward(self, t):
        return self.proj(t.unsqueeze(-1).float())

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
    
    def forward(self, x, t):
        h = F.silu(self.conv1(x))
        time_emb = F.silu(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]
        h = self.norm(h)
        return self.conv2(h)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Time embedding
        self.time_embed = TimeEmbedding(32)
        
        # Downsample path
        self.down1 = Block(1, 32, 32)
        self.down2 = Block(32, 64, 32)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        # Upsample path
        self.up1 = Block(128 + 64, 64, 32)  # Skip connection adds 64 channels
        self.up2 = Block(64 + 32, 32, 32)   # Skip connection adds 32 channels
        
        # Final layer
        self.out = nn.Conv2d(32, 1, 1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Downsample path
        h1 = self.down1(x, t_emb)          # [B, 32, 28, 28]
        h2 = self.down2(F.max_pool2d(h1, 2), t_emb)  # [B, 64, 14, 14]
        
        # Bottleneck
        h = F.max_pool2d(h2, 2)            # [B, 64, 7, 7]
        h = self.bottleneck(h)             # [B, 128, 7, 7]
        
        # Upsample path
        h = F.interpolate(h, scale_factor=2)  # [B, 128, 14, 14]
        h = self.up1(torch.cat([h, h2], dim=1), t_emb)  # [B, 64, 14, 14]
        
        h = F.interpolate(h, scale_factor=2)  # [B, 64, 28, 28]
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)  # [B, 32, 28, 28]
        
        return self.out(h)  # [B, 1, 28, 28]

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# ==============================================
# Training the model
# ==============================================
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(dataloader)
    total_loss = 0
    
    for images, _ in pbar:
        images = images.to(device)
        
        # Sampling random timesteps
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        
        # Adding noise to images
        x_noisy, noise = diffusion.add_noise(images, t)
        
        # Predict the noise
        predicted_noise = model(x_noisy, t)
        
        # Compute the loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

# ==============================================
# Sampling (Generating images)
# ==============================================
@torch.no_grad()
def sample(n_samples=16):
    """Generate images using noise"""
    model.eval()
    
    # Start with random noise
    x = torch.randn(n_samples, 1, image_size, image_size).to(device)
    
    # Denoising process (reverse diffusion)
    for t in range(timesteps-1, -1, -1):
        # Create tensor of current timestep
        t_tensor = torch.full((n_samples,), t, device=device)
        
        # Predict noise
        predicted_noise = model(x, t_tensor)
        
        # Get diffusion parameters
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alpha_bars[t]
        beta_t = diffusion.betas[t]
        
        # Calculate noise for this step
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        # Update image (reverse diffusion step)
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise
    
    # Clip to valid pixel range
    x = torch.clamp(x, -1., 1.)
    return x.cpu()

# Generate and display images
generated_images = sample()
plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(generated_images[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
plt.tight_layout()
plt.savefig('Generated_images.png')
#plt.show()

end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time}")






















