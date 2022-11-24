import fastai
#from fastai.vision.widgets import *
#from fastai.vision.all import *
from fastai import *
from fastai import load_learner

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
            st.subheader(f'Prediction: *{pred}* with Confidence: *{probs[pred_idx]*100:.02f}*%')
            #st.subheader(f'ผลการทำนาย: *{pred}* ด้วยความมั่นใจ: *{probs[pred_idx]*100:.02f}*%')
            st.subheader(f'You are currently use *{model}* model!')
    
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
            elif pred == 'Stratus' :
                st.image('St.png')
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


st.text('      ')
st.text('      ')
st.text('      ')
st.text('Credit')
st.text('[developers] Patompong Oupapong, Pannawit Wantae, Pongsapat Suporn')
st.text('[advisors] Songkran Buttawong, Suthut Butchanon')
st.text('A Capstone Project, Princess Chulabhorn Science High School Loei')
