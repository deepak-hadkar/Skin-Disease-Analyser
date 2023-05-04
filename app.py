import streamlit as st
import numpy as np
import pandas as pd
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K
import os
import time
import io
from PIL import Image
import plotly.express as px
from tensorflow import python as tf_python
import tensorflow.python as tfp 
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import streamlit.components.v1 as components
from pathlib import Path

file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

MODELSPATH = ROOT + 'models/'
DATAPATH = ROOT + 'data/'

def render_header():
    st.write("""
        <p align="center"> 
            <H1> Skin cancer Analyzer 
        </p>

    """, unsafe_allow_html=True)

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_animation_1 = "https://assets5.lottiefiles.com/packages/lf20_LmW6VioIWc.json"
predicting = "https://assets5.lottiefiles.com/private_files/lf30_jbhavmow.json"

lottie_anime_json = load_lottie_url(lottie_animation_1)
lottie_predicting_json = load_lottie_url(predicting)   

# @st.cache(allow_output_mutation=True)
def load_mekd():
    img = Image.open(DATAPATH + '/ISIC_0024312.jpg')
    return img


# @st.cache
def data_gen(x):
    img = np.asarray(Image.open(x).resize((100, 75)))
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate


# @st.cache
def data_gen_(img):
    img = img.reshape(100, 75)
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate


def load_models():

    model = load_model(MODELSPATH + "model.h5")
    return model

# @st.cache
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    ynew = model.predict_proba(x_test)
    K.clear_session()
    ynew = np.round(ynew, 2)
    ynew = ynew*100
    y_new = ynew[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    K.clear_session()
    return y_new, Y_pred_classes

# @st.cache
def display_prediction(y_new):
    """Display image and preditions from model"""

    result = pd.DataFrame({'Probability': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {2: 'Benign keratosis-like lesions', 4: 'Melanocytic nevi', 3: 'Dermatofibroma',
                        5: 'Melanoma', 6: 'Vascular lesions', 1: 'Basal cell carcinoma', 0: 'Actinic keratoses'}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result


def main():

    # Initialize the session state variable if it doesn't exist yet

    menu = ["Home", "Info", "App"]
    choice = st.sidebar.selectbox("Select a page", menu)

    if choice == "Home":

        st_lottie(lottie_anime_json, key = "hello")

        

    elif choice == "Info":
                    # front end elements of the web page 
        html_temp = """ 
        <div style ="background-color:white;padding:13px"> 
        <h1 style ="color:brown;text-align:center;font-size: 52px">Skin Disease Detection and Classification</h1> 
        </div> 
        """
        # display the front end aspect
        img1 = Image.open(ROOT + 'html_images/kjsce_header.jpeg')
        img2 = Image.open(ROOT + '/html_images/names.jpeg')
        
        st.image(img1, width=704)
        st.markdown(html_temp, unsafe_allow_html = True) 
        st.markdown("")
        st.image(img2, width=704)

    elif choice == "App":

        # Set the background color and image
        html_background = """
        <style>
        body {
            background-color: #F5F5F5;
            background-image: url('/Users/deepakhadkar/Documents/GitHub/Skin-Disease-Analyzer/html_images/background.jpg');
            background-size: cover;
            }
        </style>
        """
        st.markdown(html_background, unsafe_allow_html=True)

        st_lottie(lottie_predicting_json, speed=1, width=200, height=200, key = "hello")

        st.sidebar.header('Skin cancer Analyzer')
        st.sidebar.subheader('Choose a page to proceed:')
        page = st.sidebar.selectbox("", ["Sample Image", "Upload Your Skin Image"])

        if page == "Sample Image":
            st.header("Sample Image Prediction for Skin Cancer")
            st.markdown("""
            *Now, this is probably why you came here. Let's get you some Predictions*

            You need to choose Sample Image
            """)

            mov_base = ['Sample Image I']
            movies_chosen = st.multiselect('Choose Sample Image', mov_base)

            if len(movies_chosen) > 1:
                st.error('Please select Sample Image')
            if len(movies_chosen) == 1:
                st.success("You have selected Sample Image")
            else:
                st.info('Please select Sample Image')

            if len(movies_chosen) == 1:
                if st.checkbox('Show Sample Image'):
                    st.info("Showing Sample Image---->>>")
                    image = load_mekd()
                    st.image(image, caption='Sample Image', use_column_width=True)
                    st.subheader("Choose Training Algorithm!")
                    if st.checkbox('Keras'):
                        model = load_models()
                        st.success("Hooray !! Keras Model Loaded!")
                        if st.checkbox('Show Prediction Probablity on Sample Image'):
                            x_test = data_gen(DATAPATH + '/ISIC_0024312.jpg')
                            y_new, Y_pred_classes = predict(x_test, model)
                            result = display_prediction(y_new)
                            st.write(result)
                            if st.checkbox('Display Probability Graph'):
                                fig = px.bar(result, x="Classes",
                                            y="Probability", color='Classes')
                                st.plotly_chart(fig, use_container_width=True)

        if page == "Upload Your Skin Image":

            st.header("Upload Your Skin Image")

            file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])

            if file_path is not None:
                x_test = data_gen(file_path)
                image = Image.open(file_path)
                img_array = np.array(image)

                st.success('File Upload Success!!')
            else:
                st.info('Please upload Image file')

            if st.checkbox('Show Uploaded Image'):
                st.info("Showing Uploaded Image ---->>>")
                st.image(img_array, caption='Uploaded Image',
                        use_column_width=True)
                st.subheader("Choose Training Algorithm!")
                if st.checkbox('Keras'):
                    model = load_models()
                    st.success("Hooray !! Keras Model Loaded!")
                    if st.checkbox('Show Prediction Probablity for Uploaded Image'):
                        y_new, Y_pred_classes = predict(x_test, model)
                        result = display_prediction(y_new)
                        st.write(result)
                        if st.checkbox('Display Probability Graph'):
                            fig = px.bar(result, x="Classes",
                                        y="Probability", color='Classes')
                            st.plotly_chart(fig, use_container_width=True)


if __name__== "__main__":
    main()