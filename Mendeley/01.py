import zipfile
import os

# Pfad zur hochgeladenen ZIP-Datei
zip_path = './MendeleyDataset-20240516.zip'
extract_path = 'MendeleyDataset'

# Entpacken der ZIP-Datei
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
