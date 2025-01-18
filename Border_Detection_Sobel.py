import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt


def apply_sobel(image):
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply Sobel operator
    grad_x = signal.convolve2d(image, sobel_x, mode="same", boundary="symm")
    grad_y = signal.convolve2d(image, sobel_y, mode="same", boundary="symm")

    # Combine gradients
    sobel = np.sqrt(grad_x**2 + grad_y**2)

    return sobel


# Load the image
image = np.array(Image.open("mario1.png").convert("L"))

# Apply Sobel operator
sobel_result = apply_sobel(image)

# Normalize Sobel result
sobel_result = (
    (sobel_result - sobel_result.min())
    / (sobel_result.max() - sobel_result.min())
    * 255
)
sobel_image = Image.fromarray(sobel_result.astype(np.uint8))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display original image
ax1.imshow(image, cmap="gray")
ax1.set_title("Original Image")
ax1.axis("off")

# Display Sobel edge detection result
ax2.imshow(sobel_image, cmap="gray")
ax2.set_title("Sobel Edge Detection")
ax2.axis("off")

plt.tight_layout()
plt.show()

# Save Sobel result
sobel_image.save("sobel_edge_detection.jpg")

print("Sobel edge detection complete. Result saved as 'sobel_edge_detection.jpg'.")
