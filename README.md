# Music Genre Classification System 🎶🎧

## Overview
This project is a **Music Genre Classification System** built using **TensorFlow, Streamlit, and Librosa**. The system allows users to upload an audio file, process it, and predict the genre using a deep learning model.

## Features
- 🎵 **Upload an audio file** in `.mp3` format.
- 🎤 **Preprocessing**: Converts audio into Mel spectrograms.
- 🤖 **Deep Learning Model** trained on different music genres.
- 🎯 **Genre Prediction** using TensorFlow.
- 🖥️ **Interactive UI** with Streamlit.

## Tech Stack 🛠️
- **Python** 🐍
- **TensorFlow** for deep learning
- **Streamlit** for web UI
- **Librosa** for audio processing
- **NumPy & Matplotlib** for data processing & visualization

## Project Structure 📂
```
📁 music-genre-classification
│── 📂 Test_Music              # Folder for test audio files
│── 📂 models                  # Folder for trained models
│── app.py                     # Main Streamlit application
│── Trained_model.h5        # Trained deep learning model
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

## Usage 🎼
1. Open the **Streamlit web app**.
2. Click on **Prediction** from the sidebar.
3. **Upload an audio file** (`.mp3`).
4. Click **Play Audio** to listen.
5. Click **Predict** to classify the genre.
6. View the **predicted music genre**!

## Example Output 📊
- **Uploaded File:** `song.mp3`
- **Predicted Genre:** 🎸 `Rock`
