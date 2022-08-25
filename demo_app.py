import fastai
from fastai.vision.widgets import *
from fastai.vision.all import *

import pathlib
#pathlib.PosixPath = pathlib.WindowsPath
import streamlit as st

st.image('header2.jpg')
st.header('Clouds Classification Demonstrate')

class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload or Take a Cloud Image",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Click here! to Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.subheader(f'Prediction: {pred} with Confidence: {probs[pred_idx]*100:.02f}%')
            st.subheader(f'ผลการทำนาย: {pred} ด้วยความมั่นใจ: {probs[pred_idx]*100:.02f}%')
            st.balloons()
        #else: 
           # st.write(f'Click the button to classify') 
       

if __name__=='__main__':

    resnet_model ='CloudClassification_resnet50_v1 (1).pkl'
    predictor_resnet = Predict(resnet_model)

<<<<<<< HEAD
st.balloon()
=======
st.text('      ')
st.text('      ')
st.text('      ')
st.text('Credit')
st.text('[developers] Patompong Oupapong, Pannawit Wantae, Pongsapat Suporn')
st.text('[advisors] Songkran Buttawong, Suthut Butchanon')
st.text('A Capstone Project, Princess Chulabhorn Science High School Loei')
>>>>>>> e0840e541826ee492127e10a3fe65087b123b45a
