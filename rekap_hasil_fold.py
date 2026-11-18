import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JUMLAH_FOLD = 5
KOLOM = ["accuracy", "precision", "recall", "f1_score"]
NAMA_KELAS = ["ASLI", "AI"]   # 0 = asli, 1 = AI

baris = []
total_cm = {}

for fold in range(1, JUMLAH_FOLD + 1):
    path = f"outputs/fold{fold}/hasil_perbandingan.json"
    if not os.path.isfile(path):
        print("Fold", fold, "dilewati, file tidak ada")
        continue

    with open(path) as f:
        hasil = json.load(f)

    for r in hasil:
        baris.append({
            "model": r["model"],
            "fold": fold,
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
        })

        # jumlahkan confusion matrix semua fold
        if r["model"] not in total_cm:
            total_cm[r["model"]] = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                total_cm[r["model"]][i][j] += r["confusion_matrix"][i][j]

# hasil tiap fold
tabel = pd.DataFrame(baris).sort_values(["model", "fold"])
tabel.to_csv("outputs/rekap_tiap_fold.csv", index=False)
print("=== HASIL TIAP FOLD ===")
print(tabel.round(4).to_string(index=False))

# rata-rata semua fold
rata = tabel.groupby("model")[KOLOM].mean()
simpangan = tabel.groupby("model")[KOLOM].std()
rata.to_csv("outputs/rekap_rata_rata.csv")
print("\n=== RATA-RATA SEMUA FOLD ===")
print(rata.round(4).to_string())
print("\n=== SIMPANGAN BAKU (makin kecil makin stabil) ===")
print(simpangan.round(4).to_string())
print("\nAkurasi rata-rata tertinggi:", rata["accuracy"].idxmax())

# kelas yang sering salah
print("\n=== KESALAHAN (semua fold digabung) ===")
for model, cm in total_cm.items():
    print(model)
    print(f"  gambar ASLI ditebak AI : {cm[0][1]} dari {cm[0][0] + cm[0][1]}")
    print(f"  gambar AI ditebak ASLI : {cm[1][0]} dari {cm[1][0] + cm[1][1]}")

# gambar confusion matrix gabungan
nama_model = sorted(total_cm.keys())
fig, axes = plt.subplots(1, len(nama_model), figsize=(4.5 * len(nama_model), 4), squeeze=False)
for ax, model in zip(axes[0], nama_model):
    cm = total_cm[model]
    terbesar = max(max(cm[0]), max(cm[1]))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(NAMA_KELAS)
    ax.set_yticklabels(NAMA_KELAS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(model)
    for i in range(2):
        for j in range(2):
            warna = "white" if cm[i][j] > terbesar / 2 else "black"
            ax.text(j, i, cm[i][j], ha="center", va="center", color=warna)
fig.tight_layout()
fig.savefig("outputs/confusion_matrix_total.png", dpi=150)
print("\nGambar tersimpan di outputs/confusion_matrix_total.png")