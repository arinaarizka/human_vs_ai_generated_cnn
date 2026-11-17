import torch
import torch.nn as nn


class cnntigablok(nn.Module):
    """Algoritma 2: simple CNN yang arsitekturnya diubah dari cnnduablok.

    Perubahan dibanding cnnduablok (algoritma 1):
      - 3 blok konvolusi (sebelumnya 2)
      - kernel 3x3 + padding (sebelumnya 5x5 tanpa padding), jumlah filter 32 -> 64 -> 128
      - ada BatchNorm di tiap blok supaya training lebih stabil
      - MaxPool 2x2 (sebelumnya 3x3)
      - Global Average Pooling menggantikan flatten besar, jadi tidak bergantung ukuran gambar
      - ada Dropout sebelum lapisan akhir untuk mengurangi overfitting
    """

    def __init__(self, input_c=3, output=2, dropout=0.3):
        super(cnntigablok, self).__init__()

        # blok 1 pola sederhana (garis, tepi)
        self.convBlock1 = nn.Sequential(nn.Conv2d(in_channels=input_c, out_channels=32, kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(2, 2)))

        # blok 2 gabungin pola polanya (tekstur)
        self.convBlock2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(2, 2)))

        # blok 3 pola yang lebih rumit (bentuk, detail)
        self.convBlock3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(2, 2)))

        # rata-rata tiap peta fitur jadi 1 angka, jadi ukurannya selalu 128
        self.gap = nn.AdaptiveAvgPool2d(1)

        # human atau ai (keputusan final)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=128, out_features=output)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = torch.flatten(self.gap(x), 1)
        x = self.fc(self.dropout(x))
        return x


if __name__ == "__main__":
    model = cnntigablok()
    gambar_palsu = torch.randn(4, 3, 100, 100)
    hasil = model(gambar_palsu)
    print(model)
    print("bentuk output :", hasil.shape)
    print("jumlah parameter :", sum(p.numel() for p in model.parameters()))
