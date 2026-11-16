import torch
import torch.nn as nn

class cnnduablok(nn.Module):
    def __init__(self, input_c=3, output=2):
        super(cnnduablok, self).__init__()

        # blok 1 pola sederhana
        self.convBlock1 = nn.Sequential(nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(3, 3)))

        # blok 2 gabungin pola polanya
        self.convBlock2 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
                                                nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=(3, 3)))

        # human atau ai (keputusan final)
        self.fc = nn.Linear(in_features=50 * 9 * 9, out_features=output)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.fc(torch.flatten(x, 1))
        return x

if __name__ == "__main__":
    model = cnnduablok()
    gambar_palsu = torch.randn(4, 3, 100, 100)
    hasil = model(gambar_palsu)
    print(model)
    print("bentuk output :", hasil.shape)
    print("jumlah parameter :", sum(p.numel() for p in model.parameters()))