from PIL import Image, ImageOps
import matplotlib.pyplot as plt

img_path = "mario.png"
img = Image.open(img_path)

gray_img = img.convert("L")
fig, axes = plt.subplots(1, 3,  figsize=(12, 6))

# Afficher l'image originale
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Afficher l'image en niveaux de gris
axes[1].imshow(gray_img)
axes[1].set_title("Grayscaled Image ")
axes[1].axis("off")

axes[2].imshow(gray_img, cmap="gray")
axes[2].set_title("Grayscaled Image Corrected")
axes[2].axis("off")

plt.tight_layout()
plt.show()


"""with Image.open('11.jpg') as im:
     im.show()
     ImageOps.grayscale(im).show()"""
