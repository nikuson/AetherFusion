import aetherfusion

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
