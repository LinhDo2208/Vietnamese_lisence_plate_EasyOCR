import os, shutil, argparse

CHARS_MAC_DINH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # phai khop 'number' trong plate_config.yaml

p = argparse.ArgumentParser()
p.add_argument("--pth", default="best_accuracy.pth",
               help="duong dan file best_accuracy.pth (mac dinh: file cung thu muc)")
p.add_argument("--chars", default=CHARS_MAC_DINH,
               help="bo ky tu, dung thu tu nhu 'number' trong plate_config.yaml")
p.add_argument("--name", default="plate", help="ten model (recog_network)")
p.add_argument("--home", default=os.path.join(os.path.expanduser("~"), ".EasyOCR"),
               help="thu muc EasyOCR (mac dinh ~/.EasyOCR)")
p.add_argument("--no-test", action="store_true", help="bo qua buoc thu nap model")
args = p.parse_args()

pth = os.path.abspath(os.path.expanduser(args.pth))
if not os.path.exists(pth):
    raise SystemExit(f"Khong thay file model: {pth}")

home = args.home
os.makedirs(os.path.join(home, "model"), exist_ok=True)
os.makedirs(os.path.join(home, "user_network"), exist_ok=True)

NET = '''import torch.nn as nn


class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=256):
        super().__init__()
        oc = [int(output_channel / 8), int(output_channel / 4),
              int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, oc[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[0], oc[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[1], oc[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(oc[2], oc[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[2], oc[3], 3, 1, 1, bias=False), nn.BatchNorm2d(oc[3]), nn.ReLU(True),
            nn.Conv2d(oc[3], oc[3], 3, 1, 1, bias=False), nn.BatchNorm2d(oc[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[3], oc[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, x):
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        return self.linear(out)


class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super().__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, input, text):
        f = self.FeatureExtraction(input)
        f = self.AdaptiveAvgPool(f.permute(0, 3, 1, 2)).squeeze(3)
        f = self.SequenceModeling(f)
        return self.Prediction(f.contiguous())
'''

YAML = f"""network_params:
    input_channel: 1
    output_channel: 256
    hidden_size: 256
imgH: 64
lang_list:
    - 'en'
character_list: {args.chars}
"""

open(os.path.join(home, "user_network", f"{args.name}.py"), "w", encoding="utf-8").write(NET)
open(os.path.join(home, "user_network", f"{args.name}.yaml"), "w", encoding="utf-8").write(YAML)
shutil.copy2(pth, os.path.join(home, "model", f"{args.name}.pth"))
print("Da cai 3 file vao", home)

if not args.no_test:
    # thu nap model de kiem tra cai dat dung
    import easyocr
    easyocr.Reader(["en"], gpu=False, recog_network=args.name, detector=False,
                   model_storage_directory=os.path.join(home, "model"),
                   user_network_directory=os.path.join(home, "user_network"))
    print(f"Nap model {args.name} thanh cong")