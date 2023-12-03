import pandas as pd
import os
import glob

image_path = "../oasis/Data/"
metadata_path = "./metadata/"

label_encode = {'Non_Demented':0, 'Very_mild_Dementia':0.5, 'Moderate_Dementia':2, 'Mild_Dementia':1}

label_list = glob.glob(image_path + "*")
label_list = [label_name.split("/")[-1] for label_name in label_list]

# Export path and label in new file
image_dict = {}
for label_name in label_list:
    file_list = glob.glob(image_path + label_name + "/*")
    image_dict[label_name] = [file for file in file_list]

image_df = pd.DataFrame(columns=["id","target"])
for key,value in image_dict.items():
    new_df = pd.DataFrame(value,columns=["id"])
    new_df["target"] = label_encode[key]
    new_df["patient"] = [i.split("_mpr")[0].split("/")[-1] for i in new_df["id"]]
    image_df = pd.concat([image_df,new_df])

metadata_df = pd.read_csv(metadata_path + "oasis_cross-sectional.csv")

combined = image_df.merge(metadata_df,left_on='patient', right_on='ID')
combined.fillna("0",inplace=True)

combined.to_csv('data.csv',index=False)
