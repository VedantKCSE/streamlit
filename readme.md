# Model Repository: Download and Test Instructions

This repository provides a pre-trained model hosted on Google Drive. Follow the steps below to download and test the model.

## 📂 Model Download Link

You can download the model from the following Google Drive link:

[Download the Model](https://drive.google.com/drive/folders/1LJtAY5UnkoThj6wbawxKTHtSV_v0qnA2?usp=drive_link)

---

## 🛠️ How to Download the Model Using Python

You can automate the download process using Python. Below is a code snippet to download files from Google Drive using the `gdown` library:

```python
# Install gdown if not already installed
# pip install gdown

import gdown

# URL to the Google Drive folder
folder_url = "https://drive.google.com/drive/folders/1LJtAY5UnkoThj6wbawxKTHtSV_v0qnA2?usp=drive_link"

# Download the folder (requires folder ID extraction)
folder_id = "1LJtAY5UnkoThj6wbawxKTHtSV_v0qnA2"
gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", quiet=False)
