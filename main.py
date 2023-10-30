# imports
from audio_processing import *
from visualization import *
from utils import *

if __name__ == '__main__':
    # Replace this part with your method of uploading audio
    path = './audio/conversation.wav'   #Update this

    # Convert to wav if not wav 
    if path[-3:] != 'wav':
        path = convert_to_wav(path)

    # Transcribe the audio using Whisper ASR
    segments = transcribe_audio(path, model_size='large')

    y, sr = load_audio(path)
    duration, frames, rate = get_audio_metadata(y, sr)
    print(f"Frames: {frames}")
    print(f"Frame Rate: {rate} frames/sec")
    print(f"Duration: {duration} seconds")

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(path, duration, segment)

    embeddings = np.nan_to_num(embeddings)
    labels = cluster_speakers(embeddings, 2)  # Assuming 2 speakers for now

    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    save_transcript(segments)

    plot_2d_clusters(embeddings, segments, labels)
    plot_3d_clusters(embeddings, segments, labels)
