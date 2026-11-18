import cv2 as cv
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedKFold

# otomatis cek sedang di kaggle atau colab
if os.path.exists("/kaggle"):
    DATASET_PATH = "/kaggle/input/datasets/alessandrasala79/ai-vs-human-generated-dataset"
else:
    DATASET_PATH = "/root/.cache/kagglehub/datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4"

JUMLAH_PERKELAS = 2500
JUMLAH_FOLD = 5
SEED = 42
OUTPUT_CSV = "splits/data_5fold.csv"


def buat_data_5fold():
    df = pd.read_csv(DATASET_PATH + "/train.csv")

    # ambil 2500 gambar asli (0) dan 2500 gambar AI (1)
    asli = df[df["label"] == 0].sample(JUMLAH_PERKELAS, random_state=SEED)
    ai = df[df["label"] == 1].sample(JUMLAH_PERKELAS, random_state=SEED)
    data = pd.concat([asli, ai]).reset_index(drop=True)[["file_name", "label"]].copy()

    # beri nomor fold 
    data["fold"] = 0
    skf = StratifiedKFold(n_splits=JUMLAH_FOLD, shuffle=True, random_state=SEED)
    for nomor, (index_latih, index_uji) in enumerate(skf.split(data, data["label"]), start=1):
        data.loc[index_uji, "fold"] = nomor

    return data


if __name__ == "__main__":
    data = buat_data_5fold()

    os.makedirs("splits", exist_ok=True)
    data.to_csv(OUTPUT_CSV, index=False)

    print("Jumlah gambar:", len(data))
    print("\nJumlah per kelas (0 = asli, 1 = AI):")
    print(data["label"].value_counts().sort_index())
    print("\nJumlah per fold dan per kelas:")
    print(pd.crosstab(data["fold"], data["label"]))
    print("\nTersimpan di", OUTPUT_CSV)