import pyaudio
import numpy as np
import whisper
import torch
import threading
import queue
import time
from googletrans import Translator
import logging
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re

class PhishingDetector:
    def __init__(self, max_features=5000, max_length=100, model_path='phishing_detection_model.h5'):
        """
        Initialize Phishing Detection Model
        
        Args:
            max_features (int): Maximum vocabulary size
            max_length (int): Maximum sequence length for text
            model_path (str): Path to save/load the trained model
        """
        self.max_features = max_features
        self.max_length = max_length
        self.model_path = model_path
        
        # Tokenizer for text preprocessing
        self.tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
        
        # Label Encoder
        self.label_encoder = LabelEncoder()
        
        # Model
        self.model = None
    
    def preprocess_text(self, text):
        """
        Preprocess input text
        
        Args:
            text (str): Input text
        
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data(self, dataframe):
        """
        Prepare training data
        
        Args:
            dataframe (pd.DataFrame): Input dataset
        
        Returns:
            tuple: Processed features and labels
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in dataframe['TEXT']]
        
        # Fit tokenizer
        self.tokenizer.fit_on_texts(preprocessed_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(preprocessed_texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        
        # Encode labels
        labels = self.label_encoder.fit_transform(dataframe['LABEL'])
        labels_onehot = to_categorical(labels)
        
        return padded_sequences, labels_onehot
    
    def build_model(self, input_shape, num_classes):
        """
        Build LSTM-based neural network model
        
        Args:
            input_shape (tuple): Input shape
            num_classes (int): Number of classes
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        model = Sequential([
            Embedding(self.max_features, 128, input_length=input_shape[0]),
            LSTM(256, return_sequences=True),
            LSTM(128),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, dataframe):
        """
        Train phishing detection model
        
        Args:
            dataframe (pd.DataFrame): Training dataset
        
        Returns:
            History of model training or None
        """
        # Check if model already exists
        if os.path.exists(self.model_path):
            print(f"Existing model found at {self.model_path}. Skipping training.")
            try:
                self.model = load_model(self.model_path)
                
                # Recompile the model to ensure metrics are built
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                return None
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Proceeding with model training.")
        
        # Prepare training data
        X, y = self.prepare_data(dataframe)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        self.model = self.build_model(
            input_shape=(X_train.shape[1],), 
            num_classes=y.shape[1]
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            self.model_path, 
            save_best_only=True, 
            monitor='val_accuracy'
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"\nFinal Model Accuracy: {accuracy*100:.2f}%")
        
        return history
    
    def detect_phishing(self, text, train_data):
        """
        Detect phishing for a given text
        
        Args:
            text (str): Input text to check
            train_data (pd.DataFrame): Training dataset for preparing tokenizer
        
        Returns:
            dict: Phishing detection results
        """
        # Ensure model is loaded and tokenizer is prepared
        if self.model is None:
            print("No model found. Training a new model...")
            self.train_model(train_data)
        
        # Prepare tokenizer if needed
        if not hasattr(self.tokenizer, 'word_index') or len(self.tokenizer.word_index) == 0:
            self.prepare_data(train_data)
        
        # Preprocess input text
        preprocessed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length)
        
        # Predict
        predictions = self.model.predict(padded_sequence)[0]
        
        # Get class labels
        class_labels = self.label_encoder.classes_
        
        # Find predicted class
        predicted_class = class_labels[np.argmax(predictions)]
        prediction_prob = np.max(predictions)
        
        return {
            'original_text': text,
            'preprocessed_text': preprocessed_text,
            'predicted_label': predicted_class,
            'prediction_confidence': float(prediction_prob * 100)
        }

class RealtimeSpeechRecognition:
    def __init__(self, model_size='base', record_seconds=5, sample_rate=16000):
        """
        Initialize the real-time multilingual speech recognition system.
        
        :param model_size: Size of Whisper model (tiny, base, small, medium, large)
        :param record_seconds: Maximum duration of audio chunks to process
        :param sample_rate: Audio sample rate
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        
        # Initialize Whisper model
        logging.info(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        
        # Initialize translator
        self.translator = Translator()
        
        # Audio recording parameters
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.audio_queue = queue.Queue()
        
        # Storage for transcripts
        self.current_transcript = ""
        self.transcripts = []
        
        # PyAudio configuration
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Threading controls
        self._stop_event = threading.Event()
        self.recording_lock = threading.Lock()
        
        # Chunk size (power of 2 for efficiency)
        self.chunk = 2048
        
        # Phishing detection
        self.detector = None
        self.training_data = None
        
        # Phrase limit
        self.max_phrases = 6
        
        # Silence detection parameters
        self.silence_threshold = 0.05  # Adjust based on your environment
        self.min_phrase_duration = 1.0  # Minimum duration of a phrase
        self.max_silence_duration = 1.5  # Maximum silence before considering phrase complete

    def initialize_detector(self, training_data):
        """
        Initialize the phishing detector with training data
        """
        self.detector = PhishingDetector()
        self.training_data = training_data
        self.detector.train_model(training_data)

    def translate_text(self, text):
        """
        Translate non-English text to English
        
        :param text: Input text to translate
        :return: Translated text with original language
        """
        try:
            # Detect language
            detection = self.translator.detect(text)
            
            # If not English, translate
            if detection.lang != 'en':
                translation = self.translator.translate(text, dest='en')
                return f"{text} (Language: {detection.lang}, English: {translation.text})"
            
            return text
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function to handle incoming audio data.
        """
        if self._stop_event.is_set():
            return (None, pyaudio.paComplete)
        
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        """
        Start audio recording and recognition.
        """
        # Check if max phrases reached
        if len(self.transcripts) >= self.max_phrases:
            logging.warning("Max phrases reached.")
            return False

        # Reset stop event and clear queue
        self._stop_event.clear()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            # Open stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback
            )

            # Start recognition in a separate thread
            threading.Thread(target=self.process_audio, daemon=True).start()

            logging.info("Recording started. Speak now...")
            return True

        except Exception as e:
            logging.error(f"Recording start error: {e}")
            return False

    def process_audio(self):
        """
        Process audio chunks and transcribe using advanced phrase detection.
        """
        audio_chunks = []
        start_time = time.time()
        last_sound_time = start_time
        silence_start_time = None
        
        while not self._stop_event.is_set():
            try:
                # Get audio chunk with timeout
                chunk = self.audio_queue.get(timeout=0.5)
                audio_chunks.append(chunk)
                
                # Detect energy (volume) of the chunk
                chunk_energy = np.abs(chunk).mean()
                current_time = time.time()
                
                # Track last time sound was detected
                if chunk_energy > self.silence_threshold:
                    last_sound_time = current_time
                    silence_start_time = None
                else:
                    # Start tracking silence if not already tracking
                    if silence_start_time is None:
                        silence_start_time = current_time
                
                # Check for phrase completion conditions
                if (silence_start_time and 
                    current_time - silence_start_time > self.max_silence_duration and
                    current_time - start_time >= self.min_phrase_duration):
                    # Process the collected audio
                    if audio_chunks:
                        self.transcribe_and_process(audio_chunks)
                        # Reset for next phrase
                        audio_chunks = []
                        start_time = current_time
                        last_sound_time = current_time
                        silence_start_time = None
                
                # Stop recording if max time exceeded
                if current_time - start_time > self.record_seconds:
                    break
            
            except queue.Empty:
                # Check for prolonged silence or recording timeout
                if time.time() - last_sound_time > self.record_seconds:
                    break
        
        # Process any remaining audio
        if audio_chunks:
            self.transcribe_and_process(audio_chunks)
        
        # Signal recording completion
        self._stop_event.set()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def transcribe_and_process(self, audio_chunks):
        """
        Transcribe and process a collected audio chunk
        
        :param audio_chunks: List of audio numpy arrays
        """
        try:
            # Concatenate audio chunks
            audio_data = np.concatenate(audio_chunks)
            
            # Advanced transcription
            result = self.model.transcribe(
                audio_data, 
                fp16=torch.cuda.is_available(),
                language=None,
                condition_on_previous_text=False,
                temperature=0,
                best_of=5,
                beam_size=5
            )
            
            # Process transcription
            transcript = result['text'].strip()
            if transcript:
                # Translate 
                translated_transcript = self.translate_text(transcript)
                
                # Detect phishing if detector is initialized
                if self.detector and self.training_data is not None:
                    phishing_result = self.detector.detect_phishing(translated_transcript, self.training_data)
                    
                    # Store transcript
                    with threading.Lock():
                        self.transcripts.append(translated_transcript)
                    
                    # Print results
                    print("\n" + "="*50)
                    print(f"Transcript: {translated_transcript}")
                    print(f"Predicted Label: {phishing_result['predicted_label']}")
                    print(f"Prediction Confidence: {phishing_result['prediction_confidence']:.2f}%")
                
                # Log the transcription
                logging.info(f"🎙️ Transcribed: {translated_transcript}")
        
        except Exception as e:
            logging.error(f"Transcription error: {e}")

    def stop_recording(self):
        """
        Stop audio recording and recognition.
        """
        # Set stop event to signal threads to terminate
        self._stop_event.set()
        
        # Close stream if open
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error stopping stream: {e}")
        
        # Terminate PyAudio
        try:
            self.pyaudio_instance.terminate()
        except Exception as e:
            logging.error(f"Error terminating PyAudio: {e}")
        
        logging.info("Recording stopped.")

    def get_transcripts(self):
        """
        Retrieve all transcripts.
        
        :return: List of transcripts
        """
        return self.transcripts

def main():
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load dataset
    try:
        # Ensure the column names match exactly what's in your CSV
        df = pd.read_csv('Dataset.csv')
        
        # Verify columns exist
        required_columns = ['LABEL', 'TEXT']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Select only required columns
        df = df[required_columns]
        
        # Initialize speech recognition 
        speech_rec = RealtimeSpeechRecognition(
            model_size='base',  # More accurate model
            record_seconds=5,   # Increased recording time for complete phrases
            sample_rate=16000   # Standard high-quality sample rate
        )
        
        # Initialize phishing detector
        speech_rec.initialize_detector(df)
        
        print("🎤 Real-Time Phishing Detection")
        print("Speak up to 6 phrases. Press Ctrl+C to stop.")
        
        # Record multiple phrases
        try:
            while len(speech_rec.get_transcripts()) < speech_rec.max_phrases:
                # Start recording
                if speech_rec.start_recording():
                    # Wait for recording to complete
                    time.sleep(speech_rec.record_seconds + 0.5)
                
                # Short pause between recordings
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n🛑 Recording Interrupted")
        
        finally:
            speech_rec.stop_recording()
            
            # Final summary
            print("\n📝 Final Transcripts:")
            for i, transcript in enumerate(speech_rec.get_transcripts(), 1):
                print(f"{i}. {transcript}")
    
    except FileNotFoundError:
        print("Error: Dataset.csv not found. Please ensure the file exists.")
    except ValueError as ve:
        print(f"Dataset Error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()