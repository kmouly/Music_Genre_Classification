import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import tempfile
from matplotlib import pyplot as plt

# Cache model loading for efficiency
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("Trained_model.h5")

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        # Define the duration of each chunk and overlap
        chunk_duration = 4  # seconds
        overlap_duration = 2  # seconds
        
        # Convert durations to samples
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        
        # Calculate the number of chunks
        num_chunks = max(1, int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1)
        
        data = []
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]
            
            # Compute Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            
            # Resize to match model input shape
            mel_spectrogram_resized = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram_resized.numpy())  # Convert tensor to numpy
        
        return np.array(data)
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# Model prediction function
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
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
    "<h1 style='text-align: center; color:#FFEF00;'>Music Genre Classification System!</h1>",
    unsafe_allow_html=True
)
st.subheader("Welcome to the Music Genre Classification System! 🎶🎧")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        filepath = temp_file.name
    
    # Play the uploaded audio
    if st.button("Play Audio"):
        st.audio(filepath)
    
    # Predict the music genre
    if st.button("Predict Genre"):
        with st.spinner("Analyzing audio... Please wait..."):
            X_test = load_and_preprocess_data(filepath)
            if X_test is not None:
                result_index = model_prediction(X_test)
                st.balloons()
                
                # Genre labels
                genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                
                # Styled success message
                st.markdown(
                    f"""
                    <div style="background-color: #2196F3; padding: 10px; border-radius: 10px; 
                    color: white; font-size: 18px; text-align: center; margin: 10px 0;">
                    🎵 <b>Predicted Genre:</b> <span style='color: #FF5733;'><b>{genres[result_index]}</b></span> music!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
