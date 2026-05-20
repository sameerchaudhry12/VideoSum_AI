# 🎬 VideoSum AI: Dual-Engine Multi-Modal Video Summarizer

VideoSum AI is a professional-grade video processing application that uses Deep Learning to automatically extract the most significant moments from any video. It analyzes visual facial expressions, vocal prosody, and semantic text meaning simultaneously.

## ✨ Core Features
* **Dual-Engine Architecture:**
    * **🎬 Short Highlights Mode:** Acts as an action-camera, hunting for explosive emotional spikes to create 5-10 second viral clips.
    * **📚 Lecture Digest Mode:** Acts as an intelligent scribe. It reads complete sentences and generates a chronological 30% compression of podcasts and lectures.
* **Multi-Modal AI:**
    * **Visual:** MTCNN + MobileNetV2 (Facial Emotion Recognition)
    * **Audio:** Librosa + LSTM (Speech Tone Analysis)
    * **Text:** OpenAI Whisper + BERT (Semantic Conversational Value)
* **Instant Demo UI:** A sleek, dual-tab Gradio web application with a "Pre-Bake" engine for instantaneous live demonstrations.

## 💻 Usage
To run the web application on your local machine:
1. Clone this repository.
2. git clone https://github.com/nafayy04/VideoSum-AI.git
3. Install requirements: `pip install -r requirements.txt`
4. Run the app: `python app.py`
