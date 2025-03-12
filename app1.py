import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import tempfile

# Function to load model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)

        if len(audio_data) == 0:
            return None, "File is corrupted. Please upload a correct file."

        chunk_duration = 4  # seconds
        overlap_duration = 2  # seconds
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram)

        return np.array(data), None
    except Exception:
        return None, "File is corrupted. Please upload a correct file."


# Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    avg_probabilities = np.mean(y_pred, axis=0)  # Average probability of all chunks
    predicted_index = np.argmax(avg_probabilities)
    highest_probability = np.max(avg_probabilities)
    # Print probabilities in terminal
    print(avg_probabilities)
    return predicted_index, highest_probability

# Centered Title
st.title("🎵 Music Genre Classifier 🎶")

# Upload file
st.write("Upload an audio file and click 'Predict Genre' to identify the music genre.")
test_mp3 = st.file_uploader("", type=["mp3", "wav"])

if test_mp3 is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(test_mp3.getbuffer())
        filepath = f"Test_Music/{test_mp3.name}"
    
    with open(filepath, "wb") as f:
        f.write(test_mp3.getbuffer())
        
st.audio(test_mp3)

# Predict button with spinner beside it
col1, col2 = st.columns([3, 1])
with col1:
    predict_button = st.button("Predict Genre")
with col2:
    spinner_placeholder = st.empty()

if predict_button:
    spinner_placeholder = st.empty() 
    if test_mp3 is None:
        st.markdown(
            f"""
            <div style="background-color: #FF4C4C; 
                        padding: 10px; 
                        border-radius: 10px; 
                        color: white; 
                        font-size: 16px;
                        text-align: center;
                        margin: 10px 0;">
                ❌ <b>Error:</b> ⚠️ Please upload an audio file before clicking Predict.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        spinner_placeholder.markdown("⏳ *Processing...*")
        print("Predicting Genre")

        # Load and preprocess data
        X_test, error_message = load_and_preprocess_data(filepath)

        if X_test is None:
            spinner_placeholder.empty()
            st.markdown(
                f"""
                <div style="background-color: #FF4C4C; 
                            padding: 10px; 
                            border-radius: 10px; 
                            color: white; 
                            font-size: 16px;
                            text-align: center;
                            margin: 10px 0;">
                    ❌ <b>Error:</b> {error_message}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Model Prediction
            result_index, highest_probability = model_prediction(X_test)
            spinner_placeholder.empty()

            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            st.balloons()
            st.markdown(
                f"""
                <div style="background-color: #7C9D8E;  
                            padding: 10px; 
                            border-radius: 10px; 
                            color: white; 
                            font-size: 18px;
                            text-align: center;
                            margin: 10px 0;">
                    🎵 <b>Model Prediction:</b> It's a <span style='color: #872657;'><b>{label[result_index]}</b></span> music!
                    <br> 🎼 <b>Confidence Score:</b> {highest_probability:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )
