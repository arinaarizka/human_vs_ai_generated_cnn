"""test.py - Menguji model terlatih memakai pipeline data TIM (Utils/getData).

Data, ukuran gambar, dan pembagian fold diambil dari fungsi milik tim:
    ambil_data()  ->  getImageLabel(data)  ->  buat_fold(dataset, data["label"])
sehingga data uji persis sama dengan yang dipakai saat training (5-fold stratified, random_state=42).

Keluaran (di --out_dir, default: outputs/<model>/):
    confusion_matrix.png   confusion matrix (jumlah dari semua fold yang diuji)
    test_result.json       akurasi, presisi, recall, F1 per fold + rata-rata, parameter, waktu
    misclassified.csv      daftar gambar yang salah ditebak (bahan analisis laporan)

Contoh (jalankan dari folder utama repo, di Kaggle karena FOLDER data ada di /kaggle/input):
    python test.py --model cnnduablok  --fold 1     # uji fold 1 saja
    python test.py --model cnntigablok              # uji semua fold (butuh bobot ke-5 fold)
    python test.py --model mobilenetv2 --weights "outputs/{model}/fold{fold}.pt"

Nama file bobot (default): outputs/<model>/fold<N>.pt  -> {model} dan {fold} diisi otomatis.
Bobot dibuat oleh train.py: torch.save(model.state_dict(), path).
"""
import argparse
import csv
import importlib
import inspect
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

MODELS_DIR = "Models"
UTILS_DIR = "Utils"

# Nama class tiap model (isi folder Models). File-nya dicari otomatis di Models/*.py.
MODEL_CLASSES = {
    "cnnduablok":  "cnnduablok",         # algoritma 1: CNN 2 blok
    "cnntigablok": "cnntigablok",        # algoritma 2: CNN yang dimodifikasi
    "mobilenetv2": "MobileNetV2Model",   # algoritma 3: model dari luar (pretrained)
}

# Sesuai getData: label 0 = asli, 1 = ai
CLASS_NAMES = ["ASLI", "AI"]


def find_in_dir(folder, name):
    """Mencari class/fungsi bernama `name` di semua file folder/*.py."""
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder '{folder}' tidak ditemukan. Jalankan test.py dari folder utama repo.")
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        module = importlib.import_module(f"{folder}.{fn[:-3]}")
        if hasattr(module, name):
            return getattr(module, name)
    raise SystemExit(f"'{name}' tidak ditemukan di {folder}/*.py. Cek penulisan namanya (huruf besar/kecil berpengaruh).")


def build_model(class_name, num_classes=2):
    """Membuat model. Argumen yang tidak diterima konstruktor dilewati otomatis
    (cnnduablok memakai 'output', MobileNetV2Model memakai 'num_classes' dan 'pretrained')."""
    factory = find_in_dir(MODELS_DIR, class_name)
    wanted = {"num_classes": num_classes, "output": num_classes, "pretrained": False}
    accepted = inspect.signature(factory).parameters
    return factory(**{k: v for k, v in wanted.items() if k in accepted})


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt)


def to_nchw(x):
    """getImageLabel mengembalikan gambar (N,100,100,3); Conv2d butuh (N,3,100,100)."""
    if x.dim() == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    return x.contiguous()


def evaluate(model, subset, batch_size, device):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_pred.extend(model(to_nchw(x).to(device)).argmax(1).cpu().tolist())
            y_true.extend((y.argmax(1) if y.dim() > 1 else y).tolist())   # label one-hot -> angka
    return np.array(y_true), np.array(y_pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="nama model, mis. cnnduablok / cnntigablok / mobilenetv2")
    p.add_argument("--model_class", default=None, help="nama class di Models (kosongkan kalau sudah di MODEL_CLASSES)")
    p.add_argument("--weights", default="outputs/{model}/fold{fold}.pt",
                   help="path bobot; {model} dan {fold} diisi otomatis")
    p.add_argument("--fold", type=int, default=0, help="1-5 untuk satu fold, 0 = semua fold (default)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", default=None, help="default: outputs/<model>")
    args = p.parse_args()

    class_name = args.model_class or MODEL_CLASSES.get(args.model)
    if class_name is None:
        raise SystemExit(f"Model '{args.model}' belum ada di MODEL_CLASSES. Tambahkan atau pakai --model_class.")
    out_dir = args.out_dir or os.path.join("outputs", args.model)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data dari pipeline tim ---
    ambil_data = find_in_dir(UTILS_DIR, "ambil_data")
    getImageLabel = find_in_dir(UTILS_DIR, "getImageLabel")
    buat_fold = find_in_dir(UTILS_DIR, "buat_fold")
    data = ambil_data()
    dataset = getImageLabel(data)
    folds = buat_fold(dataset, data["label"])
    print(f"Jumlah gambar: {len(dataset)} | jumlah fold: {len(folds)}")

    fold_ids = [args.fold] if args.fold else list(range(1, len(folds) + 1))
    for k in fold_ids:
        if not 1 <= k <= len(folds):
            raise SystemExit(f"--fold harus 1 sampai {len(folds)}")
    paths = {k: args.weights.format(model=args.model, fold=k) for k in fold_ids}
    missing = [pth for pth in paths.values() if not os.path.isfile(pth)]
    if missing:
        raise SystemExit("File bobot tidak ditemukan:\n  " + "\n  ".join(missing) +
                         "\nBobot dibuat oleh train.py. Atau uji satu fold dengan --fold N.")

    per_fold, cm_total, wrong_rows, infer_time = [], np.zeros((2, 2), dtype=int), [], 0.0
    n_params = None
    for k in fold_ids:
        model = build_model(class_name).to(device)
        load_weights(model, paths[k], device)
        model.eval()
        n_params = sum(q.numel() for q in model.parameters())

        _, uji = folds[k - 1]
        t0 = time.time()
        y_true, y_pred = evaluate(model, uji, args.batch_size, device)
        infer_time += time.time() - t0

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_total += cm
        per_fold.append({"fold": k, "n_uji": int(len(y_true)), "accuracy": float(acc), "precision_macro": float(prec),
                         "recall_macro": float(rec), "f1_macro": float(f1), "confusion_matrix": cm.tolist()})
        print(f"\n=== Fold {k} | data uji: {len(y_true)} | bobot: {paths[k]} ===")
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=4, zero_division=0))

        for idx, t, pr in zip(uji.indices, y_true, y_pred):
            if t != pr:
                wrong_rows.append([k, data["file_name"][idx], CLASS_NAMES[t], CLASS_NAMES[pr]])

    # --- gabungan semua fold yang diuji ---
    def stat(key):
        v = np.array([f[key] for f in per_fold])
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    summary = {key: stat(key) for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_total, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix ({args.model}, {len(fold_ids)} fold)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm_total[i, j]), ha="center", va="center",
                    color="white" if cm_total[i, j] > cm_total.max() / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "misclassified.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "file_name", "aktual", "prediksi"])
        w.writerows(wrong_rows)

    result = {"model": args.model, "folds_tested": fold_ids, "params": n_params,
              "inference_time_sec": infer_time, "per_fold": per_fold,
              "mean_std": {k: {"mean": m, "std": s} for k, (m, s) in summary.items()},
              "confusion_matrix_total": cm_total.tolist(), "classes": CLASS_NAMES}
    with open(os.path.join(out_dir, "test_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RINGKASAN (salin ke tabel laporan) ===")
    print(f"Model        : {args.model}  | fold diuji: {fold_ids}")
    for key, label in (("accuracy", "Akurasi"), ("precision_macro", "Presisi (macro)"),
                       ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)")):
        m, s = summary[key]
        print(f"{label:<16}: {m:.4f}" + (f" ± {s:.4f}" if len(fold_ids) > 1 else ""))
    print(f"Parameter    : {n_params:,}")
    print(f"Salah tebak  : {len(wrong_rows)} gambar (lihat misclassified.csv)")
    print(f"Hasil disimpan di {out_dir}/")


if __name__ == "__main__":
    main()
