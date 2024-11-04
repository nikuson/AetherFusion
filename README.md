# AetherFusion: Diffusion Transformer U-Net for CIFAR-10 Image Generation

This repository provides a **Diffusion Transformer U-Net** model designed for image generation tasks using the CIFAR-10 dataset. The architecture combines U-Net with Transformer blocks, allowing the model to capture spatial and temporal dependencies effectively, improving image generation quality. This model leverages multi-head self-attention and label embeddings to conditionally generate images based on class labels, suitable for conditional image generation tasks.

## Features
- **U-Net Architecture**: The model is based on U-Net, enabling efficient feature extraction and spatial restoration for image synthesis.
- **Transformer Blocks**: Multi-head self-attention layers enhance the model's ability to capture complex dependencies within the image.
- **Conditional Image Generation**: The model is conditioned on CIFAR-10 class labels, which guide the image generation process.
- **Supports CIFAR-10 Dataset**: Trained on CIFAR-10, generating 32x32 images across 10 classes.

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- matplotlib

Install the necessary libraries:
```bash
pip install torch torchvision
```

## Model Architecture Overview

1. **Positional Encoding**: Used to introduce positional information to the Transformer blocks.
2. **Multi-Head Self-Attention (MHSA)**: Improves the model's ability to capture dependencies within the image.
3. **U-Net with Attention**: U-Net backbone with Transformer layers embedded in the encoder, enhancing feature representation and spatial accuracy.
4. **Label Embedding**: Embeds CIFAR-10 labels to conditionally generate images based on the input class.

## Code Description

### Model Definition
- **`PositionalEncoding`**: Adds sinusoidal positional information to each feature map.
- **`TransformerBlock`**: Standard Transformer block with multi-head self-attention and feed-forward layers, enhancing receptive field and feature extraction.
- **`UNetWithAttention`**: Core U-Net model with integrated Transformer blocks for spatially accurate feature extraction and reconstruction.
- **`DiffusionTransformerUNet`**: Main class that integrates label embeddings and U-Net with Transformer blocks for conditional image generation.

### Training
The `train` function trains the model on the CIFAR-10 dataset using a Mean Squared Error (MSE) loss, optimizing the model's weights.

### Generation
The `generate_images` function generates images conditioned on random CIFAR-10 labels, using noise as the input and progressively refining it through the network.

## Example Usage

1. **Train the Model**:
   ```python
   model = DiffusionTransformerUNet(image_size=32, num_classes=10, embedding_dim=128, num_heads=4).cuda()
   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-4)
   train(model, trainloader, criterion, optimizer, num_epochs=50)
   ```

2. **Generate Images**:
   ```python
   samples = generate_images(model, num_samples=10)
   ```

## Repository Structure

```plaintext
.
├── data                # Folder for the CIFAR-10 dataset
├── aetherfusion.py     # model
├── generate.py         # Script to generate images
└── README.md           # Documentation and usage
```

## Future Improvements

- **Experiment with Different Attention Mechanisms**: Explore other attention mechanisms, such as cross-attention, to improve conditioning.
- **Fine-tuning on Larger Datasets**: Experiment with datasets beyond CIFAR-10 for higher resolution outputs.
- **Enhanced Conditioning**: Investigate additional conditioning techniques for finer control over generated outputs.

Colab: https://colab.research.google.com/drive/1ZfRtg2kotM4FMD7QLU_kHJgqbgT5soy7?usp=sharing
---

## License
This project is licensed under the MIT License.
