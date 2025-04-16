import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import tempfile
import math

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

        # Check if the audio is silent (all zeros)
        if np.all(audio_data == 0):
            return None, "The uploaded file is silent. Please upload a valid audio file."

        # Check if the audio is too short
        if len(audio_data) == 0:
            return None, "File is corrupted. Please upload a correct file."

        chunk_duration = 4  # seconds
        overlap_duration = 2  # seconds
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

        if num_chunks <= 0:
            return None, "Audio file is too short to be processed."

        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram)

        return np.array(data), None
    except Exception as e:
        return None, f"File is corrupted. Please upload a correct file. Error: {str(e)}"

# Model Prediction with Other Genre classification
def model_prediction(X_test, confidence_threshold=0.5):
    model = load_model()
    if X_test.size == 0:
        return None, 0.0
    y_pred = model.predict(X_test)
    avg_probabilities = np.mean(y_pred, axis=0)  # Average probability of all chunks
    predicted_index = np.argmax(avg_probabilities)
    highest_probability = np.max(avg_probabilities)

    # If confidence is below the threshold, classify as "Other Genre"
    if highest_probability < confidence_threshold:
        return "Other Genre", highest_probability
    return predicted_index, highest_probability

# Streamlit UI
def main():
    # Centered Title
    st.title("🎵 Music Genre Classifier 🎶")
    
    # Description
    st.write("""
    Upload an audio file (MP3 or WAV) and click 'Predict Genre' to identify the music genre.
    If the audio doesn't clearly belong to any known genre, it will be classified as 'Other Genre'.
    """)
    
    # File uploader
    test_mp3 = st.file_uploader("", type=["mp3", "wav"])

    if test_mp3 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(test_mp3.getbuffer())
            filepath = f"Test_Music/{test_mp3.name}"
        
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())
            
        st.audio(test_mp3)

    # Predict button with spinner
    col1, col2 = st.columns([3, 1])
    with col1:
        predict_button = st.button("Predict Genre")
    with col2:
        spinner_placeholder = st.empty()

    if predict_button:
        spinner_placeholder = st.empty() 
        if test_mp3 is None:
            st.error("❌ **Error:** ⚠️ Please upload an audio file before clicking Predict.")
        else:
            with st.spinner("⏳ Processing your audio..."):
                print("Predicting Genre")

                # Load and preprocess data
                X_test, error_message = load_and_preprocess_data(filepath)

                if X_test is None:
                    spinner_placeholder.empty()
                    st.error(f"❌ **Error:** {error_message}")
                else:
                    # Model Prediction
                    result_index, highest_probability = model_prediction(X_test, confidence_threshold=0.5)
                    spinner_placeholder.empty()

                    if result_index is not None:
                        label = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                                'jazz', 'metal', 'pop', 'reggae', 'rock']
                        
                        # Custom styling for results
                        result_style = """
                            <style>
                                .prediction-box {
                                    background-color: #7C9D8E;
                                    padding: 20px;
                                    border-radius: 10px;
                                    color: white;
                                    font-size: 18px;
                                    text-align: center;
                                    margin: 20px 0;
                                }
                                .other-genre {
                                    color: #872657;
                                    font-weight: bold;
                                }
                                .known-genre {
                                    color: #FFD700;
                                    font-weight: bold;
                                }
                            </style>
                        """
                        st.markdown(result_style, unsafe_allow_html=True)

                        # If the result is "Other Genre"
                        if result_index == "Other Genre":
                            st.markdown(
                                f"""
                                <div class="prediction-box">
                                    🎵 <b>Model Prediction:</b> This is classified as <span class="other-genre">Other Genre</span>!
                                    <br><br>
                                    🎼 <b>Confidence Score:</b> {highest_probability:.2%}
                                    <br><br>
                                    <i>The audio doesn't clearly match any of the trained genres.</i>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.balloons()
                            st.markdown(
                                f"""
                                <div class="prediction-box">
                                    🎵 <b>Model Prediction:</b> This is <span class="known-genre">{label[result_index].capitalize()}</span> music!
                                    <br><br>
                                    🎼 <b>Confidence Score:</b> {highest_probability:.2%}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.error("❌ **Error:** Unable to make a prediction. Please try again with a different file.")

if __name__ == "__main__":
    main()
