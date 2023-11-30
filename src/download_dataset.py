import os
import subprocess

def download_dataset():
    # Ensure the .kaggle directory exists
    os.makedirs(os.path.join(os.path.expanduser('~'), '.kaggle'), exist_ok=True)

    # Download the dataset from Kaggle
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'ninadaithal/imagesoasis', '--unzip', '-p', 'data/'])
    
     # Confirmation message
    print("Download completed and files are extracted to the 'data/' directory.")


if __name__ == "__main__":
    download_dataset()
