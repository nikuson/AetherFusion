import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Model parameters
batch_size = 64
image_size = 32
num_classes = 10
num_epochs = 50
learning_rate = 1e-4
embedding_dim = 128
num_heads = 4

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# Multi-head self-attention block
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

# Define the improved U-Net architecture with attention
class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, num_heads):
        super(UNetWithAttention, self).__init__()
        
        # Input block
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Encoder (downsampling)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Multi-head self-attention block
        self.transformer = TransformerBlock(embedding_dim, num_heads)
        
        # Decoder (upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Output block
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Pass through encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        
        # Apply transformer block
        transformer_input = enc2.flatten(2).permute(2, 0, 1)
        transformer_output = self.transformer(transformer_input)
        enc2 = transformer_output.permute(1, 2, 0).view(enc2.size())
        
        # Decoder
        dec1 = self.dec1(enc2)
        
        # Combine with original data and apply final output block
        out = self.final(dec1 + enc1)
        
        return out

# Embedding and U-Net integration
class DiffusionTransformerUNet(nn.Module):
    def __init__(self, image_size, num_classes, embedding_dim, num_heads):
        super(DiffusionTransformerUNet, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.position_encoding = PositionalEncoding(embedding_dim)
        self.unet = UNetWithAttention(3, 3, embedding_dim, num_heads)
        
        # Linear transformation to match input image size
        self.label_projection = nn.Linear(embedding_dim, 3)

    def forward(self, x, labels):
        # Get label embedding and project to match input image channels (3)
        label_embedding = self.embedding(labels)  # (batch_size, embedding_dim)
        label_embedding = self.label_projection(label_embedding).unsqueeze(-1).unsqueeze(-1)
        
        # Expand label embedding to image size
        label_embedding = label_embedding.expand(-1, -1, x.size(2), x.size(3))
        
        # Add label embedding to input image
        x = x + label_embedding
        return self.unet(x)

# Initialize model, loss function, and optimizer
model = DiffusionTransformerUNet(image_size, num_classes, embedding_dim, num_heads).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, trainloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs, labels)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")

# Start training
train(model, trainloader, criterion, optimizer, num_epochs)

# Generate images function
def generate_images(model, num_samples=10):
    model.eval()
    samples = []
    for _ in range(num_samples):
        label = torch.randint(0, num_classes, (1,)).cuda()
        noise = torch.randn(1, 3, image_size, image_size).cuda()
        with torch.no_grad():
            generated_image = model(noise, label).cpu()
        samples.append(generated_image)
    return samples

# Function to display images
def show_images(images, num_images=10, cols=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(num_images // cols + 1, cols, i + 1)
        image = images[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Denormalize for display
        image = (image * 0.5 + 0.5)  # Bringing values to range [0, 1]
        
        # Clip values to prevent warnings
        image = np.clip(image, 0, 1)  # Limit values in range [0, 1]
        
        plt.imshow(image)
        plt.axis('off')
    plt.show()

# Generate and display images
samples = generate_images(model, num_samples=10)
show_images(samples)
