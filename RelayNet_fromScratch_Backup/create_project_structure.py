import os

dirs = [
    "models",
    "scripts",
    "utils",
    "data"
]

files = [
    "models/simple_cnn.py",
    "utils/data_loader.py",
    "scripts/train.py",
    "scripts/test.py",
    "main.py",
    "requirements.txt",
    "data/__init__.py",
    "models/__init__.py",
    "scripts/__init__.py",
    "utils/__init__.py"
]

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for file in files:
    with open(file, 'w') as f:
        pass  # Erstellen einer leeren Datei
