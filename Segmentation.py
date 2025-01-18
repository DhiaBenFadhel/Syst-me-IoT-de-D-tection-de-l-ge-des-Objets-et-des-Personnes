import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve

# Charger l'image et la convertir en niveaux de gris
image_path = "mario1.png"  # Remplacez par le chemin de votre image
image = Image.open(image_path).convert("L")

# Convertir l'image en tableau numpy
image_array = np.array(image)

# Noyaux de Sobel pour la détection des contours
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Appliquer les noyaux
edges_x = convolve(image_array, sobel_x)
edges_y = convolve(image_array, sobel_y)

# Combiner les résultats pour obtenir l'intensité des contours
edges = np.hypot(edges_x, edges_y)
edges = (edges / edges.max()) * 255  # Normaliser les valeurs

# Seuillage simple
threshold = 128
segmentation = (image_array > threshold) * 255

# Afficher les images originales et traitées
fig, axes = plt.subplots(1, 4, figsize=(20, 15))

# Image originale
axes[0].imshow(image_array, cmap="gray")
axes[0].set_title("Image Originale")
axes[0].axis("off")

# Détection de contours
axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Contours (Sobel)")
axes[1].axis("off")

# Segmentation par seuillage
axes[2].imshow(segmentation, cmap="gray")
axes[2].set_title("Segmentation par Seuillage")
axes[2].axis("off")

# Affichage combiné (segmentation + contours)
combined = np.maximum(segmentation, edges)
axes[3].imshow(combined, cmap="gray")
axes[3].set_title("Contours + Segmentation")
axes[3].axis("off")

plt.tight_layout()
plt.show()
