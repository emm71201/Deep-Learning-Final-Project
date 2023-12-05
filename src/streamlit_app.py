import streamlit as st
import os
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import process_demos as mydemos

# model = demos.model
# test_ds = demos.test_ds


st.set_option('deprecation.showPyplotGlobalUse', False)
#PATH = "../oasis/train"
PATH = "data/Data"



classmap = {"Mild Dementia":0, "Moderate Dementia":1, "No Dementia":2,\
            "Very Mild Dementia":3}
class_names = os.listdir(PATH)
class_names.sort()

gmlp_citation = "@misc{liu2021pay,\
    title={Pay Attention to MLPs}, author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},\
    year={2021}, eprint={2105.08050},\
    archivePrefix={arXiv},\
    primaryClass={cs.LG}\
}"
def insert_citation(bibtex_entry):
    citation = f"[@{bibtex_entry.split('{')[1][:-1]}]"  # Extracting the citation key from BibTeX entry
    st.markdown(citation)
def show_image(classname='random', filename='random'):
    if classname == 'random':
        classname = random.choice(os.listdir(PATH))
    if filename == 'random':
        filename = random.choice(os.listdir(os.path.join(PATH, classname)))

    image_path = os.path.join(PATH, classname, filename)
    fig, ax = plt.subplots()
    try:
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_title(f"{classname.replace('_', ' ')}")
        ax.axis('off')
    except:
        print("The image does not exist")
    return fig


st.title("Predicting Alzheimer’s Disease with Deep Learning")
st.subheader("Authors: Nina Ebensperger, Karina Martinez, Edison Murairi")

st.header("Problem Statement")
st.text("We aim to leverage the OSAIS MRI dataset to develop a machine learning model capable \
of predicting AD onset before clinical symptoms are apparent.")

st.header("Exploratory Data Analysis")
st.subheader("View Images")

col1, col2, col3 = st.columns([1,1,1])
fig = show_image(class_names[0])
row1 = fig
with col3:
    view_image_class1 = st.button("View Image from Mild Dementia", type="primary")
    if view_image_class1:
        classname = class_names[0]
        fig = show_image(classname=classname)
        row1 = fig

# with col2:
#     view_image_class2 = st.button("View Image from Moderate Dementia", type="primary")
#     if view_image_class2:
#         classname = class_names[1]
#         fig = show_image(classname=classname)
#         row1 = fig

with col1:
    view_image_class3 = st.button("View Image from No Dementia", type="primary")
    if view_image_class3:
        classname = class_names[2]
        fig = show_image(classname=classname)
        row1 = fig

with col2:
    view_image_class4 = st.button("View Image from Very Mild Dementia", type="primary")
    if view_image_class4:
        classname = class_names[3]
        fig = show_image(classname=classname)
        row1 = fig

st.pyplot(row1)

st.subheader("Data Distribution")

col1, col2, col3, col4,col5 = st.columns([1,1,1,1,1])
with col1:
    barplot = st.button("Show Class Distribution", type="primary")
with col2:
    genderdist = st.button("Show distribution by gender", type="primary")
with col3:
    agedist = st.button("Show distribution by age", type="primary")
with col4:
    educdist = st.button("Show distribution by education", type="primary")
with col5:
    sesdist = st.button("Show distribution by socio-economic", type="primary")

if barplot:
    st.image("Figures/bar_plot.png", caption="Number of patients for each class")
elif genderdist:
    st.image("Figures/gender_dist.png", caption="Number of patients by gender")
elif agedist:
    st.image("Figures/age_dist.png", caption="Number of patients by age")
elif educdist:
    st.image("Figures/educ_dist.png", caption="Number of patients by education")
elif sesdist:
    st.image("Figures/ses.png", caption="Number of patients by socio-economic level")
else:
    st.image("Figures/bar_plot.png", caption="Number of patients for each class")

col1, col2, col3, col4,col5 = st.columns([1,1,1,1,1])
with col1:
    mmse = st.button("Show distribution by MMSE", type="primary")
with col2:
    etiv = st.button("Show distribution by eTIV", type="primary")
with col3:
    nwbv = st.button("Show distribution by NWBV", type="primary")
with col4:
    asf = st.button("Show distribution by ASF", type="primary")

if mmse:
    st.image("Figures/mmse_dist.png", caption="Number of patients by MMSE")
elif etiv:
    st.image("Figures/etiv_dist.png", caption="Number of patients by eTIV")
elif nwbv:
    st.image("Figures/nWBV_dist.png", caption="Number of patients by NWBV")
elif asf:
    st.image("Figures/asf_dist.png", caption="Number of patients by ASF")

gmlp_link = "https://arxiv.org/abs/2105.08050"
gmlp_link_markdown = f'<a href="{gmlp_link}" target="_blank">GMLP</a>'

st.header("Modeling")
st.subheader("Gating Multilayer Perceptron (gMLP)")
st.markdown(gmlp_link_markdown, unsafe_allow_html=True)

st.markdown("- GMLP is a variant of multilayer perceptrons introduced in the paper linked above.")
st.markdown("- It is an MLP with special layers called gating layers.")
st.markdown("- Convilutional neural networks perform well but have many parameters")
st.markdown("- GMLP achieves similar performance but with modest number of parameters.")

st.image("Figures/gmlp_arc.png", caption="GMLP architecture from arXiv:2105.08050")

st.image("Figures/gmlp_accuracy_loss.png", caption="GMLP accuracy and loss")
st.image("Figures/gmlp_cm.png", caption="GMLP confusion matrix")
def run_demo1():
    image, labels = mydemos.get_demo("gmlp_demo")
    st.session_state['image'] = image
    st.session_state['labels'] = labels

col1, col2, col3  = st.columns([1,1,1])

with col1:
    rundemo1 = st.button("Run Demo", on_click=run_demo1, type = "primary")
with col2:
    correct_label1 = st.button("Correct Label", type="primary")
with col3:
    model_label1 = st.button("Model Label", type="primary")
if 'image' in st.session_state:
    st.image(st.session_state['image'])
    if correct_label1:
        st.text(f"Correct Label: {st.session_state['labels'][0]}")
    if model_label1:
        st.text(f"Correct Label: {st.session_state['labels'][0]}")
        st.text(f"Model   Label: {st.session_state['labels'][1]}")




st.subheader("RESNET 50 with Metadata")

st.markdown("- Images were trained on the ResNet50 architecture")
st.markdown("- Output of convolution blocks concatenated with tabular metadata")
st.markdown("- Combined features trained with MLP to generated classification output")

st.image("Figures/resnet50_model_accuracy_loss.png", caption="Accuracy and loss for ResNet 50 with metadata")
st.image("Figures/resnet50_model_cm.png", caption="Confusion matrix for ResNet 50 with metadata")

def run_demo2():
    image, labels = mydemos.get_demo("gmlp_demo")
    st.session_state['image2'] = image
    st.session_state['labels2'] = labels


col1, col2, col3  = st.columns([1,1,1])

with col1:
    rundemo2 = st.button("Run Demo 2", on_click=run_demo2, type = "primary")
with col2:
    correct_label2 = st.button("Correct Label 2", type="primary")
with col3:
    model_label2 = st.button("Model Label 2", type="primary")
if 'image2' in st.session_state:
    st.image(st.session_state['image2'])
    if correct_label2:
        st.text(f"Correct Label: {st.session_state['labels2'][0]}")
    if model_label2:
        st.text(f"Correct Label: {st.session_state['labels2'][0]}")
        st.text(f"Model   Label: {st.session_state['labels2'][1]}")

st.subheader("CNN with Attention: ")
st.markdown("**CNN with CBAM Overview**")
st.markdown("- Enhances feature focus and model interpretability in image analysis tasks. CBAM's Role in CNNs")
st.markdown("- Channel Attention: Prioritizes 'what' features to focus on by weighting channels.")
st.markdown("- Spatial Attention: Identifies 'where' to focus, emphasizing critical regions.")
st.markdown("**Grad-CAM for Visualization**")
st.markdown("- Produces heatmaps that highlight influential areas affecting the model's decisions. Facilitates understanding of the model's internal reasoning process. Importance for Explainable AI (XAI)")
st.markdown("- Offers transparency in model predictions, crucial for trust in sensitive applications like healthcare.")
st.markdown("- Assists in model validation and iterative refinement by aligning predictions with domain expertise.")

st.image("Figures/attentioncnn_model_accuracy_loss.png", caption="Accuracy and loss for CNN with Attention Model")
st.image("Figures/attentioncnn_model_cm.png", caption="Confusion matrix for CNN with Attention")

def run_demo3():
    image, labels = mydemos.get_demo("gmlp_demo")
    st.session_state['image3'] = image
    st.session_state['labels3'] = labels

col1, col2, col3  = st.columns([1,1,1])

with col1:
    rundemo3 = st.button("Run Demo 3", on_click=run_demo3, type = "primary")
with col2:
    correct_label3 = st.button("Correct Label 3", type="primary")
with col3:
    model_label3 = st.button("Model Label 3", type="primary")
if 'image3' in st.session_state:
    st.image(st.session_state['image3'])
    if correct_label3:
        st.text(f"Correct Label: {st.session_state['labels3'][0]}")
    if model_label3:
        st.text(f"Correct Label: {st.session_state['labels3'][0]}")
        st.text(f"Model   Label: {st.session_state['labels3'][1]}")

cbam = st.button("Show XAI", type = "primary")
if cbam:
    st.image("Figures/attentioncnn_model_xai.png", caption="Explanability of CNN with Attention")

st.subheader("Future Work and Conclusion")

st.markdown("**Generate more images of the very mild dementia and mild dementia classes**")
st.markdown("* The dataset has more images of the no dementia class compared to the other classes")
st.markdown("* This in-balance skews the model towards predicting often that there is no dementia")
st.markdown("* The model learns the features associated with no dementia but not how to distinguish between no dementia and other classes.")
st.markdown("* For example, the confusion matrices show that the model does not distinguish well between no dementia and very mild dementia, making early diagnosis challeging")

st.markdown("**Implement 3D models**")
st.markdown("* The brain images are 3-Dimensional")
st.markdown("* We have used each slice of the brain as an individual image, ignoring 3D spatial informations.")
st.markdown("* The spatial relations between different pixels may be necessary in diagnosing AD.")
st.markdown("* We may consider models such as:")
st.markdown("   1. 3D Convolutional Neural Networks")
st.markdown("   2. 3D Reccurent Visual Attention Networks (RVN)")
st.markdown("* The state of the art is ~90% achieved by an RVN and a Transformer model in arXiv:2011.14139")

st.markdown("**In conclusion**")
st.markdown("We have used GMLP, ResNet Model with metadata and CNN with Attention to diagnose Alzheimer's disease")
st.markdown("We established modest results ~0.50% accuracy but promising with some improvements.")
st.markdown("One improvement is to generate data of under-represented classes artificially.")
st.markdown("Another possible improvement is to use 3D models to also account for the spatial propoerties of the brain more accurately.")
st.markdown("Thank you!")