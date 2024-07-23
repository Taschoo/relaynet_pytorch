import os

# Verzeichnisstruktur
dirs = [
    "models",
    "scripts",
    "utils",
    "saved_models"
]

# Leere Dateien, die erstellt werden sollen
files = [
    "models/blocks.py",
    "models/simple_cnn.py",
    "scripts/train.py",
    "scripts/test.py",
    "utils/data_loader.py",
    "main.py",
    "requirements.txt",
    "data/__init__.py",
    "models/__init__.py",
    "scripts/__init__.py",
    "utils/__init__.py"
]

# Verzeichnisse erstellen
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

# Leere Dateien erstellen
for file in files:
    with open(file, 'w') as f:
        pass  # Erstellen einer leeren Datei

print("Verzeichnisstruktur und leere Dateien wurden erfolgreich erstellt.")
