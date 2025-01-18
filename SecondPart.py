import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def convolve2D(img, noyau):

    height, width = img.shape
    n_height, n_width = noyau.shape

    out_height = height - n_height + 1
    out_width = width - n_width + 1

    output = np.zeros((out_height, out_width))

    for y in range(out_height):
        for x in range(out_width):
            # G(x,y) = Σ Σ F(x-i, y-j) * H(i,j)
            output[y, x] = np.sum(img[y : y + n_height, x : x + n_width] * noyau)

    return output


img = Image.open("mario.png")
img_matrix = np.array(img.convert("L"))

# lissage or Box blur kernel
noyau = np.ones((3, 3)) / 9
# Gaussian Blur Kernel
gaussian_blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
# Détection des bords or Edge Detection Kernel
edge_detection = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


result1 = convolve2D(img_matrix, noyau)
result1 = (result1 - result1.min()) / (result1.max() - result1.min()) * 255
output_img1 = Image.fromarray(result1.astype(np.uint8))

result2 = convolve2D(img_matrix, gaussian_blur)
result2 = (result2 - result2.min()) / (result2.max() - result2.min()) * 255
output_img2 = Image.fromarray(result2.astype(np.uint8))

result3 = convolve2D(img_matrix, edge_detection)
result3 = (result3 - result3.min()) / (result3.max() - result3.min()) * 255
output_img3 = Image.fromarray(result3.astype(np.uint8))

fig, axes = plt.subplots(1, 4, figsize=(12, 6))

# Afficher l'image originale
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Afficher Box Blur Convolved Image
axes[1].set_title("Box Blur Convolved Image")
axes[1].imshow(output_img1, cmap="gray")
axes[1].axis("off")

# Afficher Gaussian Blur Convolved Image
axes[2].set_title("Gaussian Blur Convolved Image")
axes[2].imshow(output_img2, cmap="gray")
axes[2].axis("off")

# Afficher Edge Detection Convolved Image
axes[3].set_title("Edge Detection Convolved Image")
axes[3].imshow(output_img3, cmap="gray")
axes[3].axis("off")

plt.tight_layout()
plt.show()
