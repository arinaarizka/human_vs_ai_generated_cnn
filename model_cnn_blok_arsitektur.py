import torch
import torch.nn as nn

def buat_model(jumlah_kelas=2):
    model = nn.Sequential(# blok 1
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool2d(2),
                          # blok 2 
                          nn.Conv2d(32, 64, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool2d(2),
                          # keputusan
                          nn.Flatten(),
                          nn.Linear(64 * 32 * 32, 128),
                          nn.ReLU(),
                          nn.Linear(128, jumlah_kelas)
                          )
    return model

if __name__ == "__main__":
    model = buat_model()
    gambar_palsu = torch.randn(4, 3, 128, 128)
    hasil = model(gambar_palsu)
    print(model)
    print("bentuk output :", hasil.shape)