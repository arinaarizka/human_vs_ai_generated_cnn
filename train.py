import os
import copy
import gc
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Models.model_cnn_blok_arsitektur import cnnduablok
from Models.model_cnn_modifikasi import cnntigablok
from Models.MobileNetV2 import MobileNetV2Model

#rules
if os.path.exists("/kaggle"):
    DATASET_PATH = "/kaggle/input/datasets/alessandrasala79/ai-vs-human-generated-dataset"
else:
    DATASET_PATH = "/root/.cache/kagglehub/datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4"

DATA_CSV = "splits/data_5fold.csv"
JUMLAH_FOLD = 5   

BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints"   # diganti tiap fold
os.makedirs("checkpoints", exist_ok=True)

print("Device:", DEVICE)

#transformasi gambar

transform_cnn = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

transform_mobilenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#dataset

class AIDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = os.path.join(
            DATASET_PATH,
            row["file_name"]
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return image, label

#membaca data

data_semua = pd.read_csv(DATA_CSV)  

print("Jumlah data seluruhnya:", len(data_semua))
print(data_semua["label"].value_counts().sort_index())


#fungsi training

def train_model(model, train_loader, val_loader, model_name):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    best_accuracy = -1
    best_state = copy.deepcopy(model.state_dict())

    history = []

    for epoch in range(EPOCHS):

        #training
        model.train()

        train_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        #validation
        model.eval()

        val_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = loss_function(outputs, labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(
                    predictions.cpu().tolist()
                )

                all_labels.extend(
                    labels.cpu().tolist()
                )

        val_loss /= len(val_loader)

        accuracy = accuracy_score(
            all_labels, all_predictions
        )

        precision = precision_score(
            all_labels, all_predictions,
            zero_division=0
        )

        recall = recall_score(
            all_labels, all_predictions,
            zero_division=0
        )

        f1 = f1_score(
            all_labels, all_predictions,
            zero_division=0
        )

        print(
            f"{model_name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Accuracy: {accuracy:.4f}"
        )

        history.append({
            "model": model_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # simpan model terbaik
        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_state,
                "model_name": model_name,
                "epoch": epoch + 1,
                "accuracy": accuracy
            }, f"{CHECKPOINT_DIR}/{model_name}_best.pt")   # DIUBAH: folder per fold

    model.load_state_dict(best_state)

    print(
        f"\nModel terbaik {model_name}: "
        f"{best_accuracy:.4f}"
    )

    return history

#training ketiga model, diulang untuk tiap fold 

all_history = []

for fold in range(1, JUMLAH_FOLD + 1):

    # fold ini jadi data uji, fold lainnya jadi data latih
    train_df = data_semua[data_semua["fold"] != fold]
    val_df = data_semua[data_semua["fold"] == fold]

    # folder model dan file data uji per fold
    CHECKPOINT_DIR = f"checkpoints/fold{fold}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    val_df.to_csv(f"splits/test_fold{fold}.csv", index=False)

    print("\n" + "#" * 60)
    print(f"FOLD {fold} DARI {JUMLAH_FOLD}")
    print("Jumlah data training:", len(train_df))
    print("Jumlah data validation:", len(val_df))
    print("#" * 60)

    #konfigurasi model (dipindah ke sini supaya model dibuat baru tiap fold)

    model_configs = [

        {
            "name": "CNN_Dua_Blok",
            "model": cnnduablok(input_c=3, output=2),
            "transform": transform_cnn
        },

        {
            "name": "CNN_Tiga_Blok",
            "model": cnntigablok(input_c=3, output=2),
            "transform": transform_cnn
        },

        {
            "name": "MobileNetV2",
            "model": MobileNetV2Model(
                num_classes=2,
                pretrained=True,
                freeze_features=False
            ),
            "transform": transform_mobilenet
        }

    ]

    history_fold = []

    for config in model_configs:

        model_name = config["name"]

        print("\n" + "=" * 60)
        print(f"FOLD {fold} | MEMULAI TRAINING:", model_name)
        print("=" * 60)

        train_dataset = AIDataset(
            train_df,
            config["transform"]
        )

        val_dataset = AIDataset(
            val_df,
            config["transform"]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        history = train_model(
            config["model"],
            train_loader,
            val_loader,
            model_name
        )

        # catat nomor fold di tiap baris riwayat
        for baris in history:
            baris["fold"] = fold

        history_fold.extend(history)

        del config["model"]
        del train_dataset, val_dataset
        del train_loader, val_loader

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # simpan riwayat tiap selesai satu fold
    all_history.extend(history_fold)
    pd.DataFrame(history_fold).to_csv(f"training_results_fold{fold}.csv", index=False)
    pd.DataFrame(all_history).to_csv("training_results.csv", index=False)

#simpan hasil

history_df = pd.DataFrame(all_history)

print("\n" + "=" * 60)
print("SEMUA TRAINING SELESAI!")
print("=" * 60)

print("\nHASIL TERBAIK SETIAP MODEL DI TIAP FOLD:")   # DIUBAH

best_results = (
    history_df.sort_values("accuracy", ascending=False)
    .groupby(["model", "fold"])
    .head(1)
    .sort_values(["model", "fold"])
)

print(best_results.to_string(index=False))

print("\nRATA-RATA 5 FOLD:") 
print(
    best_results.groupby("model")[["accuracy", "precision", "recall", "f1_score"]]
    .mean().round(4).to_string()
)

print("\nCheckpoint tersimpan di folder checkpoints/fold1 sampai fold5")
print("Hasil training tersimpan di training_results.csv")