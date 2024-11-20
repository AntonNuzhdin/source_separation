import os
import shutil

import gdown

URL_DPRNN_MODEL = "https://drive.google.com/file/d/1t9Is5QYhNjJFPK5tnZ5TVQPK_UfuEg1D/view?usp=drive_link"
URL_CONVTASNET_MODEL = "https://drive.google.com/uc?id=1jFtw5W3znI6P9EBAgfwrzHcYtnNFDImW"


def download():
    gdown.download(URL_DPRNN_MODEL)
    gdown.download(URL_CONVTASNET_MODEL)

    os.makedirs("src/convtasnet", exist_ok=True)
    os.makedirs("src/dprnn", exist_ok=True)
    shutil.move("dprnn_weights.pth", "src/dprnn/dprnn_weights.pth")
    shutil.move("convtasnet_weights.pth", "src/convtasnet/convtasnet_weights.pth")


if __name__ == "__main__":
    download()