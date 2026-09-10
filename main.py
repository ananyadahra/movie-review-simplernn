import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

import json

with open("word_index.json", "r") as f:
    word_index = json.load(f)

reversed_word_index = {value: key for (key, value) in word_index.items()}


model=load_model('simplernn_imdb.h5')
    
### preprocess user input
def preprocess_input(text):
    words= text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    encoded_review = [idx if idx<10000 else 2 for idx in encoded_review]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def decode_review(encoded_review):
    return ''.join([reversed_word_index.get(i-3,'?') for i in encoded_review])

### predict sentiment function
def predict_sentiment(review):
    preprocessed=preprocess_input(review)
    prediction=model.predict(preprocessed)
    sentiment= "positive" if prediction[0][0]>0.5 else "negative"
    return sentiment, prediction[0][0]


##stramlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("This app predicts the sentiment of a movie review as positive or negative")
 ## user input
user_input= st.text_area("Enter your movie review here:")
if(st.button('classify')):
   sentiment,prediction=predict_sentiment(user_input)
   st.write(f"sentiment:{sentiment}")
   st.write(f"likability:{prediction}")
else:
    st.write("Please enter a movie review and click classify to see the sentiment prediction.")   

   
