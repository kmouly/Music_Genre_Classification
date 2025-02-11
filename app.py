import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
import numpy as np
from tensorflow.image import resize

#Function
@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model("Trained_model.h5")
  return model


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)



#Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]


# Set background image
def set_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image URL
set_bg("https://thumbs.dreamstime.com/b/beautiful-wallpaper-music-notes-341785800.jpg") 
# Centered Title
st.markdown(
    "<h1 style='text-align: center; color:#FFEF00 ;'>Music Genre Classification System!</h1>", 
    unsafe_allow_html=True
)
st.subheader("Welcome to the Music Genre Classification System! 🎶🎧")

test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])
if test_mp3 is not None:
    filepath = f"Test_Music/{test_mp3.name}"
    
    # Save the file
    with open(filepath, "wb") as f:
        f.write(test_mp3.getbuffer())

#Show Button
if(st.button("Play Audio")):
    st.audio(test_mp3)
    
#Predict Button
if(st.button("Predict")):
    with st.spinner("Please Wait.."):       
        X_test = load_and_preprocess_data(filepath)
        result_index = model_prediction(X_test)
        st.balloons()
        label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    # Styled Success Message for Prediction
        st.markdown(
            f"""
            <div style="
                background-color: #2196F3;  /* Blue background */
                padding: 10px; 
                border-radius: 10px; 
                color: white; 
                font-size: 18px;
                text-align: center;
                margin: 10px 0;">
                🎵 <b>Model Prediction:</b> It's a <span style='color: #FF5733;'><b>{label[result_index]}</b></span> music!
            </div>
            """,
            unsafe_allow_html=True
        )
