from PIL import Image
import os

def get_unique_pixel_values(image_path):
    """
    Gibt die eindeutigen Pixelwerte eines Bildes zurück.

    :param image_path: Pfad zum Bild
    :return: Set von eindeutigen Pixelwerten
    """
    if not os.path.exists(image_path):
        print(f"Bild nicht gefunden: {image_path}")
        return set()
    else:
        # Bild laden
        image = Image.open(image_path)

        # Bild in ein Pixel-Array umwandeln
        pixels = list(image.getdata())

        # Eindeutige Pixelwerte sammeln
        unique_pixels = set(pixels)

        return unique_pixels

# Beispielverwendung
mask_path = 'data/h_1/masks/mask_1.png'  # Pfad zu Ihrem Bild
image_path = 'data/h_1/original.jpg'  # Pfad zu Ihrem Bild
unique_pixels = get_unique_pixel_values(mask_path)

# Eindeutige Pixelwerte anzeigen
print("Eindeutige Pixelwerte im Bild:")
for pixel in unique_pixels:
    print(pixel)
unique_pixels = get_unique_pixel_values(image_path)

# Eindeutige Pixelwerte anzeigen
print("Eindeutige Pixelwerte im Bild:")
for pixel in unique_pixels:
    print(pixel)
