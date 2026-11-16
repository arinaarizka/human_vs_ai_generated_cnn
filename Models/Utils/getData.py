import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedKFold

FOLDER = "/kaggle/input/ai-vs-humas-generated-dataset"
CSV = FOLDER + "/train.csv"
JUMLAH_PERKELAS = 2500

def ambil_data():
    df = pd.read_csv(CSV)
    asli = df[df["label"] == 0].sample(JUMLAH_PERKELAS, random_state=42)
    ai = df[df["label"] == 01].sample(JUMLAH_PERKELAS, random_state=42)
    data = pd.concat([asli, ai]).reset_index(drop=True)

    print("jumlah data per kelas (0=asli, 1=ai)")
    print(data["label"].value_counts())
    return data

class getImageLabel(Dataset):
    def __init__(self, data):
        self.dataset = []
        to_one_hot = np.eye(2) # jumlah kelas = 2
        for i in range(len(data)):
            nama_file = data["file name"][i]
            label = data["label"][i]

            # baca gambarnya
            image = cv.reasize(cv.imread(FOLDER + "/" + nama_file), (100, 100)) / 255
            self.dataset.append([image, to_one_hot[label]])

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtpe=torch.float32)

    def __len__(self):
        return len(self.dataset)

def buat_fold(dataset, label):
    # bagi data jd 5 bgian
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hasil = []
    for index_latih, index_uji in skf.split(label, label):
        hasil.append([Subset(dataset, index_latih), Subset(dataset, index_uji)])
    return hasil

if __name__ == "__main__":
    data = ambil_data()
    dataset = getImageLabel(data)
    print("jumlah gambar :", len(dataset))
    for nomor, (latih, uji) in enumerate(buat_fold(dataset, data["label"]), start=1):
        print("Fold", nomor, ": latih =", len(latih), ", uji =", len(uji))