import fastai
from fastai.vision.widgets import *
from fastai.vision.all import *

import pathlib
#pathlib.PosixPath = pathlib.WindowsPath
import streamlit as st

st.image('Header.png')
st.header('Clouds Classification Demonstrate')

model = st.radio(
    'What MODEL do you want to use?',
    ('Resnet50', 'MobileNet v3 small', 'VGG16 with batch normalization')
)
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
            st.markdown(
            """
            <text style="font-size: 60px;font-family:'Sarabun', sans-serif;font-weight: bold;">Prediction: *{pred}* with Confidence: *{probs[pred_idx]*100:.02f}*%</text>
            <text style="font-size: 30px;font-family: 'Sarabun', sans-serif;font-weight: bold;">ผลการทำนาย: *{pred}* ด้วยความมั่นใจ: *{probs[pred_idx]*100:.02f}*%</text><br>
            <text style="font-size: 30px;font-family: 'Sarabun', sans-serif;font-weight: bold;padding:0px 0px 0px 60px">You are currently use *{model}* model!</text>
            """
            ,unsafe_allow_html=True)
    
            if pred == 'Cirrus' :
                st.image('Ci.png')
            elif pred == 'Cirrocumulus' :
                st.image('Cc.png')
            elif pred == 'Cirrostratus' :
                st.image('Cs.png')
            elif pred == 'Altostratus' :
                st.image('As.png')
            elif pred == 'Altocumulus' :
                st.image('Ac.png')
            elif pred == 'Nimbostratus' :
                st.image('Nb.png')
            elif pred == 'Cumulus' :
                st.image('Cu.png')
            elif pred == 'Cumulonimbus' :
                st.image('Cb.png')
            elif pred == 'Cumulus' :
                st.image('Cu.png')
            else :
                st.image('Sc.png')

            st.balloons()
        #else: 
           # st.write(f'Click the button to classify') 


if model == 'Resnet50':
    if __name__=='__main__':
        #st.write('You are currently use Resnet50 model')
        resnet_model ='CloudClassification_resnet50_v1 (1).pkl'
        predictor_resnet = Predict(resnet_model)
elif model == 'MobileNet v3 small':
    if __name__=='__main__':
        #st.write('You are currently use MobilenetV3 model')
        mobilenet_model ='CloudClassification_mobilenetv3_v1.pkl'
        predictor_mobilenet = Predict(mobilenet_model)
else:
    if __name__=='__main__':
        #st.write('You are currently use VGG16 model')
        vgg16_model ='CloudClassification_vgg16_v2.pkl'
        predictor_vgg16 = Predict(vgg16_model)

def sidebar_about():
    with st.sidebar:
        st.markdown('<bighead>About this project</bighead>', unsafe_allow_html=True)
        a,b,c,d,e,f = st.columns(6)
        st.write('Organized by')
        a.image('Non.png', caption='Patompomg Oupapong')
        b.image('Kongtap.png', caption='Pannawit Wantae')
        c.image('F.png', caption='Pongsapat Suporn')
        d.image('Non.png', caption='Songkran Buttawong')
        e.image('Non.png', caption='Suthut Butchanon')
        st.markdown('<text style=\'font-size:14px;\'>A capstone project</a> Princess Culabhorn Science High School Loei</text>',unsafe_allow_html=True)


        st.markdown('<bighead>Contact</bighead>', unsafe_allow_html=True)
        st.markdown('<text>pannawit04149@pcshsloei.ac.th</text>', unsafe_allow_html=True)
sidebar_about()
"""
st.text('      ')
st.text('      ')
st.text('      ')
st.text('Credit')
st.text('[developers] Patompong Oupapong, Pannawit Wantae, Pongsapat Suporn')
st.text('[advisors] Songkran Buttawong, Suthut Butchanon')
st.text('A Capstone Project, Princess Chulabhorn Science High School Loei')
"""