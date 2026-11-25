import streamlit as st 
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# loading the tensorflow model for prediction
model = load_model('model3.h5')

# loadig tokenizer 
with open('Tokenizer.pkl', 'rb') as f:
    tk = pickle.load(f)

st.title('Twitter tweets Sentiment Analysis')

tweet = st.text_area('Enter the tweet:  ')

if st.button('Predict Sentiment') and tweet.strip():
    sequence = tk.texts_to_sequences([tweet])
    sequence = pad_sequences(sequence, padding='post', maxlen=99) # this is the user entered tweet whose total length of words
    # should match with the same amount of words/values which we have in our predicted modeel which is 99, so post padding will 
    # provide number of zeros at teh end of the actual words. 
    
    prediction = model.predict(sequence) # this pull the predicted resut from the model as per entered tweet and provide result
    predicted_class = np.argmax(prediction, axis=1)[0] # this does the job of pulling out max value from above given 3 result
    # and pull the col index of the max value (axis(1) is for col and 0 is for row) and provide teh result in 1d array hence
    # at teh end of the code you have to mention [0] to pull that only single digit from 1d array into proper single int. 
    
    sentiment_map = {0:'Negative', 1:'Neutral', 2:'Positive'} # this is simple dict
    
    st.write("Sentiment", sentiment_map[predicted_class]) # this will take the index number from above predicted_class and 
    # pulls out it value as per its given key from above dict. 
    
