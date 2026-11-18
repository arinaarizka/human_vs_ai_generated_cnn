"""test_data.py - Menguji ketiga model hasil train.py pada DATA UJI TERPISAH, lalu membandingkannya.
 
Mengikuti train.py tim:
  - checkpoint     : checkpoints/<nama_model>_best.pt   (dibuat oleh train.py)
  - praproses gambar SAMA dengan train.py (CNN: 100x100; MobileNetV2: 224x224 + normalisasi ImageNet)
  - data uji       : CSV berisi kolom file_name dan label (0 = asli, 1 = AI), TIDAK dipakai saat training
 
Contoh (dari folder utama repo):
    python test_data.py --dataset_path <folder dataset> --test_csv <test_split.csv>
    python test_data.py --models CNN_Tiga_Blok          # satu model saja
 
Keluaran (di --out_dir, default: outputs/):
    hasil_perbandingan.csv / .json   akurasi, presisi, recall, F1, parameter, waktu tiap model
    confusion_matrix.png             confusion matrix ketiga model
    perbandingan_metrik.png          grafik batang perbandingan
    training.png                     kurva training (kalau training_results.csv ada)
    misclassified_<model>.csv        gambar yang salah ditebak (bahan pembahasan)
 
Presisi, recall, dan F1 dihitung untuk kelas AI (label 1) sebagai positif, sama seperti train.py.
"""
import argparse
import json
import os
import time
 
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
 
from Models.model_cnn_blok_arsitektur import cnnduablok
from Models.model_cnn_modifikasi import cnntigablok
from Models.MobileNetV2 import MobileNetV2Model
 
# Praproses: HARUS sama persis dengan train.py
transform_cnn = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])
 
transform_mobilenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
CLASS_NAMES = ["ASLI", "AI"]   # label 0 = asli, 1 = ai
 
# Nama model HARUS sama dengan "name" di model_configs pada train.py
MODEL_SPECS = {
    "CNN_Dua_Blok":  (lambda: cnnduablok(input_c=3, output=2), transform_cnn),
    "CNN_Tiga_Blok": (lambda: cnntigablok(input_c=3, output=2), transform_cnn),
    "MobileNetV2":   (lambda: MobileNetV2Model(num_classes=2, pretrained=False, freeze_features=False),
                      transform_mobilenet),
}
 
 
class UjiDataset(Dataset):
    def __init__(self, dataframe, dataset_path, transform):
        self.data = dataframe.reset_index(drop=True)
        self.dataset_path = dataset_path
        self.transform = transform
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = Image.open(os.path.join(self.dataset_path, row["file_name"])).convert("RGB")
        return self.transform(image), int(row["label"])
 
 
def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    info = {k: ckpt[k] for k in ("epoch", "accuracy") if isinstance(ckpt, dict) and k in ckpt}
    return info
 
 
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)
 
 
def summarize(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_uji": int(len(y_true)),
        "salah_tebak": int((y_true != y_pred).sum()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
 
 
def plot_confusion(results, path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        ax.set_title(f"{r['model']}\nakurasi {r['accuracy']:.4f}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_comparison(results, path):
    metrics = [("accuracy", "Akurasi"), ("precision", "Presisi"), ("recall", "Recall"), ("f1_score", "F1")]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in enumerate(results):
        vals = [r[k] for k, _ in metrics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=r["model"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics])
    low = min(min(r[k] for k, _ in metrics) for r in results)
    ax.set_ylim(max(0, low - 0.1), 1.1)
    ax.set_title("Perbandingan model pada data uji")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_training(history_csv, path):
    if not os.path.isfile(history_csv):
        print(f"[lewati] {history_csv} tidak ada, training.png tidak dibuat")
        return
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in df.groupby("model"):
        ax[0].plot(g["epoch"], g["train_loss"], label=f"{name} (train)")
        ax[0].plot(g["epoch"], g["val_loss"], "--", label=f"{name} (val)")
        ax[1].plot(g["epoch"], g["accuracy"], label=name)
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(fontsize=7)
    ax[1].set_title("Akurasi validasi"); ax[1].set_xlabel("Epoch"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="/kaggle/input/datasets/alessandrasala79/ai-vs-human-generated-dataset",
                   help="folder dataset (berisi train_data/...)")
    p.add_argument("--test_csv", default="/kaggle/working/splits/test_split.csv",
                   help="CSV data uji: kolom file_name dan label")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--history_csv", default="training_results.csv")
    p.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
 
    test_df = pd.read_csv(args.test_csv)
    print(f"Data uji: {len(test_df)} gambar | per kelas:\n{test_df['label'].value_counts().sort_index().to_string()}")
 
    results = []
    for name in args.models:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{name}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"\n[lewati] {name}: {ckpt_path} tidak ditemukan (jalankan train.py dulu)")
            continue
        build, transform = MODEL_SPECS[name]
        model = build().to(device)
        info = load_checkpoint(model, ckpt_path, device)
        n_params = sum(q.numel() for q in model.parameters())
 
        loader = DataLoader(UjiDataset(test_df, args.dataset_path, transform),
                            batch_size=args.batch_size, shuffle=False, num_workers=2)
        t0 = time.time()
        y_true, y_pred = predict(model, loader, device)
        waktu = time.time() - t0
 
        r = summarize(name, y_true, y_pred)
        r.update({"parameter": int(n_params), "waktu_uji_detik": round(waktu, 2),
                  "epoch_terbaik": info.get("epoch"), "akurasi_val_saat_training": info.get("accuracy")})
        results.append(r)
 
        print(f"\n=== {name} | bobot: {ckpt_path} | epoch terbaik: {info.get('epoch')} ===")
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                    digits=4, zero_division=0))
        salah = test_df.reset_index(drop=True).loc[y_true != y_pred, ["file_name"]].copy()
        salah["aktual"] = [CLASS_NAMES[i] for i in y_true[y_true != y_pred]]
        salah["prediksi"] = [CLASS_NAMES[i] for i in y_pred[y_true != y_pred]]
        salah.to_csv(os.path.join(args.out_dir, f"misclassified_{name}.csv"), index=False)
 
        del model, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    if not results:
        raise SystemExit("Tidak ada model yang bisa diuji. Pastikan train.py sudah dijalankan dan folder checkpoints ada.")
 
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results])
    table = table.sort_values("accuracy", ascending=False).reset_index(drop=True)
    table.to_csv(os.path.join(args.out_dir, "hasil_perbandingan.csv"), index=False)
    with open(os.path.join(args.out_dir, "hasil_perbandingan.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_confusion(results, os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_comparison(results, os.path.join(args.out_dir, "perbandingan_metrik.png"))
    plot_training(args.history_csv, os.path.join(args.out_dir, "training.png"))
 
    print("\n=== PERBANDINGAN (salin ke tabel laporan) ===")
    show = table[["model", "accuracy", "precision", "recall", "f1_score", "parameter", "waktu_uji_detik"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = table.iloc[0]
    print(f"\nAkurasi tertinggi pada data uji: {best['model']} ({best['accuracy']:.4f}) dari {best['n_uji']} gambar")
    if len(table) > 1 and (table.loc[0, "accuracy"] - table.loc[1, "accuracy"]) < 0.02:
        print("Catatan: selisih akurasi dua teratas < 2 poin. Dengan data uji sekecil ini selisih itu "
              "bisa kebetulan, jangan disimpulkan mutlak.")
    print(f"Hasil disimpan di {args.out_dir}/")
 
 
if __name__ == "__main__":
    main()
