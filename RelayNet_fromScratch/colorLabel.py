import os
from PIL import Image
import numpy as np
import utils.colors as colors  # Assuming the colors.py file is in the same directory

def print_colored(text, color):
    print(f"{getattr(colors, color, '')}{text}{colors.RESET}")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Clear the console
os.system('cls' if os.name == 'nt' else 'clear')
print_colored("COLORING LABELS", 'BG_WHITE')

# Define paths
base_dirs = ['data/759x568', 'data/951x456']

# Define colors (as RGB tuples) for each label index
label_colors_hex = {
    1: "#731300",   # Dunkles Rotbraun
    2: "#8e2c00",   # Dunkelrot
    3: "#a84900",   # Kupfer
    4: "#c75b00",   # Warmes Orange
    5: "#e0703b",   # Sanftes Orange
    6: "#b86c7f",   # Mauve
    7: "#5b5b9e",   # Grau-Blau
    8: "#004373"    # Tiefes Blau
}
label_colors = {label: hex_to_rgb(color) for label, color in label_colors_hex.items()}

for base_dir in base_dirs:
    labels_dir = os.path.join(base_dir, 'labels')
    colored_labels_dir = os.path.join(base_dir, 'labels_colored')

    # Ensure source directory exists
    if not os.path.exists(labels_dir):
        print_colored(f"Directory '{labels_dir}' not found.", 'RED')
        continue

    # Create destination directory if it doesn't exist
    os.makedirs(colored_labels_dir, exist_ok=True)

    # List all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        try:
            with Image.open(label_path) as label_img:
                label_array = np.array(label_img)
                # Set the background to white
                color_image = np.full((label_array.shape[0], label_array.shape[1], 3), (255, 255, 255), dtype=np.uint8)
                
                for label, color in label_colors.items():
                    color_image[label_array == label] = color

                colored_label_path = os.path.join(colored_labels_dir, label_file)
                Image.fromarray(color_image).save(colored_label_path)
                print_colored(f"Colored label created at '{colored_label_path}'", 'GREEN')
        except Exception as e:
            print_colored(f"Error processing '{label_path}': {e}", 'RED')
