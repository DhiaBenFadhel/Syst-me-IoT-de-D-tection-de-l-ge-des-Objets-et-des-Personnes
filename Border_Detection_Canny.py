import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt


def canny_edge_detection(image, low_threshold, high_threshold):
    # Step 1: Noise reduction
    blurred = ndimage.gaussian_filter(image, sigma=1.0)

    # Step 2: Gradient calculation
    gradient_x = ndimage.sobel(blurred, axis=0)
    gradient_y = ndimage.sobel(blurred, axis=1)

    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Step 3: Non-maximum suppression
    height, width = gradient_magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.uint8)

    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]

    # Step 4: Double thresholding
    thresholded = np.zeros((height, width), dtype=np.uint8)
    thresholded[suppressed >= high_threshold] = 255
    thresholded[(suppressed <= high_threshold) & (suppressed >= low_threshold)] = 128

    # Step 5: Edge tracking by hysteresis
    final_edges = np.zeros((height, width), dtype=np.uint8)
    final_edges[thresholded == 255] = 255

    weak_x, weak_y = np.where(thresholded == 128)
    for i, j in zip(weak_x, weak_y):
        if 255 in thresholded[i - 1 : i + 2, j - 1 : j + 2]:
            final_edges[i, j] = 255

    return final_edges


# Load the image
image = np.array(Image.open("mario1.png").convert("L"))

# Apply Canny edge detection
canny_result = canny_edge_detection(image, low_threshold=30, high_threshold=100)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display original image
ax1.imshow(image, cmap="gray")
ax1.set_title("Original Image")
ax1.axis("off")

# Display Canny edge detection result
ax2.imshow(canny_result, cmap="gray")
ax2.set_title("Canny Edge Detection")
ax2.axis("off")

plt.tight_layout()
plt.show()

# Save Canny result
canny_image = Image.fromarray(canny_result)
canny_image.save("canny_edge_detection.jpg")

print("Canny edge detection complete. Result saved as 'canny_edge_detection.jpg'.")
