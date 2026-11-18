
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

TRAIN_CSV = "splits/train_split.csv"
VAL_CSV = "splits/val_split.csv"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Jumlah data training:", len(train_df))
print("Jumlah data validation:", len(val_df))


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
            }, f"checkpoints/{model_name}_best.pt")

    model.load_state_dict(best_state)

    print(
        f"\nModel terbaik {model_name}: "
        f"{best_accuracy:.4f}"
    )

    return history

#konfigurasi model

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

#training ketiga model

all_history = []

for config in model_configs:

    model_name = config["name"]

    print("\n" + "=" * 60)
    print("MEMULAI TRAINING:", model_name)
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

    all_history.extend(history)

    del config["model"]
    del train_dataset, val_dataset
    del train_loader, val_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

#simpan hasil

history_df = pd.DataFrame(all_history)

history_df.to_csv(
    "training_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("SEMUA TRAINING SELESAI!")
print("=" * 60)

print("\nHASIL TERBAIK SETIAP MODEL:")

best_results = (
    history_df.sort_values("accuracy", ascending=False)
    .groupby("model")
    .head(1)
)

print(best_results.to_string(index=False))

print("\nCheckpoint tersimpan di folder checkpoints/")
print("Hasil training tersimpan di training_results.csv")
