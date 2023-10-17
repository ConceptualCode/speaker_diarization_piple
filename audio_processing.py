import whisper
import subprocess
import torch
import pyannote.audio
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
import librosa
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Initialize the embedding model
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"))

def convert_to_wav(input_path):
    subprocess.call(['ffmpeg', '-i', input_path, 'audio.wav', '-y'])
    return 'audio.wav'

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def get_audio_metadata(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    frames = len(y)
    rate = sr
    return duration, frames, rate

def transcribe_audio(path, model_size='large'):
    model = whisper.load_model(model_size)
    result = model.transcribe(path)
    segments = result["segments"]
    return segments

def segment_embedding(path, duration, segment):
    audio = Audio()
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def cluster_speakers(embeddings, num_speakers):
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    return labels