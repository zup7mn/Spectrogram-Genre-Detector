
def spectrogram_of_middle(song_path, sr=41100, n_fft=8192):

    """
    Compute the spectrogram of the middle 20 seconds of a song.

    Parameters:
    song_path (str): Path to the audio file.
    sr (int): Sample rate for loading the audio file.
    n_fft (int): Number of FFT bins.
    hop_length (int): Number of samples between successive bins.
    
    Returns:
    numpy.ndarray: a spectrogram of the middle 20 seconds of the song.
    """    
    
    import librosa
    import torchaudio.transforms as T
    import torch
    import numpy as np

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
    spectrogram = T.Spectrogram(n_fft=n_fft)(torch.tensor(segment).float())
    
    image = librosa.power_to_db(spectrogram.numpy())

    return image


def save_spec_array_to_file(spec, genre, db="MusicDB", labels_file_name="labels.csv"):
    """
    Save the spectrogram array to a file with a unique hash id.
    
    Parameters:
    spec (numpy.ndarray): The spectrogram array to save.
    genre (str): The genre of the song.
    db (str): The database directory where the file will be saved.
    labels_file_name (str): The CSV file where the labels will be saved.

    Returns:
    str: The filename of the saved spectrogram.
    """
    import numpy as np
    import os
    import hashlib
    
    # Create a unique hash for the spectrogram
    hash_id = hashlib.md5(spec.tobytes()).hexdigest()
    
    # Create the directory if it doesn't exist
    os.makedirs(db, exist_ok=True)
    
    # Define the filename
    filename = f"{db}/{hash_id}.npy"
    
    # Save the spectrogram array to a .npy file
    np.save(filename, spec)

    # Append the genre to the labels CSV file
    labels_file_path = os.path.join(db, labels_file_name)
    with open(labels_file_path, 'a') as f:
        f.write(f"{hash_id},{genre}\n")
    
    return filename