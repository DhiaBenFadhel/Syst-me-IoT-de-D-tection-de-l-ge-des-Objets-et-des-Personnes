import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal


def convolution_downsampling(image, kernel_size):
    # Créer un noyau de moyenne
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    # Appliquer la convolution
    convolved = signal.convolve2d(image, kernel, mode="valid")

    # Sous-échantillonner l'image convoluée
    return convolved[::kernel_size, ::kernel_size]


# Charger l'image
original_image = Image.open("11.jpg")
original_array = np.array(original_image.convert("L"))  # Convertir en niveaux de gris

# Définir le facteur de réduction
scale_factor = 4

# Réduction par convolution
conv_downsampled = convolution_downsampling(original_array, scale_factor)

# Réduction par rééchantillonnage bilinéaire
bilinear_downsampled = np.array(
    original_image.resize(
        (original_image.width // scale_factor, original_image.height // scale_factor),
        Image.BILINEAR,
    )
)

# Réduction par rééchantillonnage bicubique
bicubic_downsampled = np.array(
    original_image.resize(
        (original_image.width // scale_factor, original_image.height // scale_factor),
        Image.BICUBIC,
    )
)

# Afficher les résultats
fig, axes = plt.subplots(1, 4, figsize=(30, 20))

axes[0].imshow(original_image)
axes[0].set_title("Image originale")
axes[0].axis("off")

axes[1].imshow(conv_downsampled, cmap="gray")
axes[1].set_title("Réduction par convolution")
axes[1].axis("off")

axes[2].imshow(bilinear_downsampled, cmap="gray")
axes[2].set_title("Réduction bilinéaire")
axes[2].axis("off")

axes[3].imshow(bicubic_downsampled, cmap="gray")
axes[3].set_title("Réduction bicubique")
axes[3].axis("off")

plt.tight_layout()
plt.show()

# Sauvegarder les images réduites
Image.fromarray(conv_downsampled).save("convolution_downsampled.jpg")
Image.fromarray(bilinear_downsampled).save("bilinear_downsampled.jpg")
Image.fromarray(bicubic_downsampled).save("bicubic_downsampled.jpg")
