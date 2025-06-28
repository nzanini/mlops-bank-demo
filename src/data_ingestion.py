import os
import zipfile
import urllib.request
import pandas as pd

RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# URL del ZIP
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00222/bank-additional.zip"
)

def download_and_extract(url: str, out_dir: str):
    zip_path = os.path.join(out_dir, "bank.zip")
    # Descargar el ZIP manualmente
    print("Downloading ZIP...")
    urllib.request.urlretrieve(url, zip_path)
    print("ZIP downloaded.")

    # Extraer todo
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    print("ZIP extracted.")

    # Leer el CSV que queremos
    csv_path = os.path.join(out_dir, "bank-additional", "bank-additional-full.csv")
    df = pd.read_csv(csv_path, sep=";")

    # Guardar en data/raw/bank.csv
    out_csv = os.path.join(out_dir, "bank.csv")
    df.to_csv(out_csv, index=False)
    print(f"Raw data saved to {out_csv}")

if __name__ == "__main__":
    download_and_extract(DATA_URL, RAW_DATA_DIR)
