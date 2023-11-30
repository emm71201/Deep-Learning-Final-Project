import pandas as pd
import os
import glob

image_path = "../oasis/Data/"
metadata_path = "./metadata/"

label_encode = {'Non Demented':0, 'Very mild Dementia':0.5, 'Moderate Dementia':2, 'Mild Dementia':1}

label_list = glob.glob(image_path + "*")
label_list = [label_name.split("/")[-1] for label_name in label_list]

image_dict = {}

for label_name in label_list:
    file_list = glob.glob(image_path + label_name + "/*")
    image_dict[label_name] = [file.split("_mpr")[0].split("/")[-1] for file in file_list]

image_df = pd.DataFrame(columns=["Patient","Code"])
for key,value in image_dict.items():
    new_df = pd.DataFrame(value,columns=["Patient"])
    new_df["Code"] = label_encode[key]
    image_df = pd.concat([image_df,new_df])

metadata_df = pd.read_csv(metadata_path + "oasis_cross-sectional.csv")

combined = image_df.merge(metadata_df[['ID','CDR']],left_on='Patient', right_on='ID')
combined.fillna("0",inplace=True)
combined['Code'] = combined['Code'].astype(float)
combined['CDR'] = combined['CDR'].astype(float)

issue_ids = combined[combined['Code'] != combined['CDR']] # All image folders match patient metadata
issue_ids