import os
import subprocess
import glob
import pandas as pd
# try:
#     import gdown
# except:
#     subprocess.run(['pip', 'install', 'gdown'])
#     import gdown

def download_dataset():
    # Ensure the .kaggle directory exists
    os.makedirs(os.path.join(os.path.expanduser('~'), '.kaggle'), exist_ok=True)

    # Download the dataset from Kaggle
    try:
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'ninadaithal/imagesoasis', '--unzip', '-p', 'data/'])
    except:
        print("Failed to download from Kaggle")
        print("This can be fixed by setting up the kaggle.json authentication in .kaggle")
        print("Please, contact the author if the issue persists")
        print("Exiting...")
        return

    # Update subfolder names
    name_dict = {'Non Demented': 'Non_Demented',
                 'Very mild Dementia': 'Very_mild_Dementia',
                 'Mild Dementia': 'Mild_Dementia',
                 'Moderate Dementia': 'Moderate_Dementia'}

    for key, value in name_dict.items():
        subprocess.run(['mv', f'data/Data/{key}', f'data/Data/{value}'])

    # Confirmation message
    print("Download completed and files are extracted to the 'data/' directory.")


def download_metadata():
    # Create metadata folder
    directory = 'metadata'
    OR_PATH = os.getcwd()
    path = os.path.join(OR_PATH, directory)
    os.mkdir(path)

    # Download file to metadata folder
    os.chdir(path)
    subprocess.run(['wget', 'https://oasis-brains.org/files/oasis_cross-sectional.csv'])

    # Return to original folder
    os.chdir(OR_PATH)

    print("Download completed and metadata files extracted to the metadata/ directory")


def make_data_file():
    image_path = "./data/Data/"
    metadata_path = "./metadata/"

    label_encode = {'Non_Demented': 0, 'Very_mild_Dementia': 0.5, 'Moderate_Dementia': 2, 'Mild_Dementia': 1}

    label_list = glob.glob(image_path + "*")
    label_list = [label_name.split("/")[-1] for label_name in label_list]

    # Export path and label in new file
    image_dict = {}
    for label_name in label_list:
        file_list = glob.glob(image_path + label_name + "/*")
        image_dict[label_name] = [file for file in file_list]

    image_df = pd.DataFrame(columns=["id", "target"])
    for key, value in image_dict.items():
        new_df = pd.DataFrame(value, columns=["id"])
        new_df["target"] = label_encode[key]
        new_df["patient"] = [i.split("_mpr")[0].split("/")[-1] for i in new_df["id"]]
        image_df = pd.concat([image_df, new_df])

    metadata_df = pd.read_csv(metadata_path + "oasis_cross-sectional.csv")

    combined = image_df.merge(metadata_df, left_on='patient', right_on='ID')
    combined.fillna("0", inplace=True)

    combined.to_csv('data.csv', index=False)

if __name__ == "__main__":

    download_dataset()
#    download_metadata()
#    make_data_file()
