import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal


def apply_convolution(img, kernel):
    # Use scipy's convolve2d function
    result = signal.convolve2d(img, kernel, mode="valid")

    # Normalize the result
    result = (result - result.min()) / (result.max() - result.min()) * 255
    return Image.fromarray(result.astype(np.uint8))


# Load the image
img = Image.open("mario.png")
img_matrix = np.array(img.convert("L"))

# Define kernels
box_blur = np.ones((3, 3)) / 9
gaussian_blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
edge_detection = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Apply convolutions
output_img1 = apply_convolution(img_matrix, box_blur)
output_img2 = apply_convolution(img_matrix, gaussian_blur)
output_img3 = apply_convolution(img_matrix, edge_detection)

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Display images
images = [img, output_img1, output_img2, output_img3]
titles = ["Original Image", "Box Blur", "Gaussian Blur", "Edge Detection"]

for ax, image, title in zip(axes, images, titles):
    if title == "Original Image":
        ax.imshow(image)
    else:
        ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
