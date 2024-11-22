import os
import shutil

import gdown


URL_CONVTASNET = "https://drive.google.com/uc?id=1u6mfea6viHbnPaOmRxFgVaa2s2R3s_gA"
URL_DPRNN = "https://drive.google.com/uc?id=1t9Is5QYhNjJFPK5tnZ5TVQPK_UfuEg1D"


def download():
    gdown.download(URL_DPRNN)
    gdown.download(URL_CONVTASNET)

    os.makedirs("src/weights", exist_ok=True)
    shutil.move("dprnn_weights.pth", "src/weights/dprnn_weights.pth")
    shutil.move("convtasnet.pth", "src/weights/convtasnet.pth")


if __name__ == "__main__":
    download()
