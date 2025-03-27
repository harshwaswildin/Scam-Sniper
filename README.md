# Scam-Sniper
Snipe out voice based phishing attempts using AI (Glitchcon 2025) 
# Real-Time Phishing Detection Speech Recognition System 🎤🕵️

## Overview

This project is an advanced real-time speech recognition and phishing detection system that leverages machine learning to analyze spoken text for potential phishing attempts. By combining speech-to-text technology, language translation, and a custom neural network-based phishing detector, the system provides an innovative approach to identifying potential security risks in spoken communication.

## Features 🌟

- **Real-Time Speech Recognition**
  - Uses OpenAI Whisper for multilingual speech-to-text transcription
  - Supports advanced audio chunk processing
  - Dynamic silence detection for accurate phrase segmentation

- **Multilingual Translation**
  - Automatic detection and translation of non-English speech
  - Preserves original text and provides English translation

- **Phishing Detection**
  - LSTM-based neural network for text classification
  - Preprocessing and tokenization of input text
  - Confidence scoring for phishing prediction

- **Flexible Configuration**
  - Configurable Whisper model size
  - Customizable recording parameters
  - Adaptable to different training datasets

## Prerequisites 🛠️

### Hardware
- GPU recommended for optimal performance
- Minimum 8GB RAM
- Multi-core processor

### Software
- Python 3.8+
- PyTorch
- TensorFlow
- Whisper
- PyAudio
- NumPy
- Pandas
- scikit-learn
- googletrans

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/yourusername/realtime-phishing-detection.git
cd realtime-phishing-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Prepare your training dataset (`Dataset.csv`)
   - Must have 'LABEL' and 'TEXT' columns
   - Label column should contain binary/multi-class phishing classifications
   - Text column contains corresponding text samples

2. Run the application:
```bash
python phishing_detection_speech_recognition.py
```

3. Speak into your microphone
   - System will transcribe and analyze up to 6 phrases
   - Press Ctrl+C to stop recording

## Dataset Requirements 📊

Your `Dataset.csv` should have the following structure:
- Column `LABEL`: Classification labels (e.g., 'phishing', 'legitimate')
- Column `TEXT`: Corresponding text samples

## Technical Details 🔬

### Speech Recognition
- Uses Whisper for multilingual transcription
- Configurable model sizes (tiny to large)
- Advanced audio chunk processing

### Phishing Detection
- LSTM neural network architecture
- Text preprocessing with tokenization
- Supports multi-class classification
- Confidence-based predictions

## Model Training 📈

- Training occurs automatically if no pre-trained model exists
- Supports transfer learning
- Early stopping and model checkpointing
- 80/20 train-test split with stratification

## Logging and Output 📋

- Real-time transcription logging
- Phishing detection results with confidence scores
- Multilingual translation support

## Limitations ⚠️

- Requires quality microphone input
- Accuracy depends on training dataset
- Performance varies with background noise

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

Distributed under the MIT License. See `LICENSE` for more information.

## Contact 📧

Your Name - youremail@example.com

Project Link: [https://github.com/yourusername/realtime-phishing-detection](https://github.com/yourusername/realtime-phishing-detection)

---

**Disclaimer**: This tool is for educational and research purposes. Always use responsibly and in compliance with ethical guidelines.
