import librosa
import torchaudio.transforms as T
import torch
import numpy as np

def spectrogram_of_middle(song_path, sr=2**13, n_fft=2**11, hop_length=2**10):

    # Load the audio file, convert it to mono, downsample to sr
    y, sr = librosa.load(song_path, sr=sr, mono=True)

    # make sure the song is long enough

    if len(y)//sr < 20:
        raise ValueError("The song is too short to analyze. It must be at least 20 seconds long.")
    


    # Calculate the middle index
    middle_index = len(y) // 2
    
    segment = y[middle_index - sr*10:middle_index + sr*10]  # 10 seconds segment around the middle
    
    
    print(segment.shape)

    # Compute the spectrogram
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)(torch.tensor(segment).float())
    
    image = librosa.power_to_db(spectrogram.numpy())

    return torch.from_numpy(image)