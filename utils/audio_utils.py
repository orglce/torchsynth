import os
import shutil
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import torchaudio.functional

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interprete

if isnotebook(): 
    import IPython.display as ipd
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
else:
    class IPD:
        def Audio(*args, **kwargs):
            pass

        def display(*args, **kwargs):
            pass
    ipd = IPD()

def calc_stft(waveforms, device, sample_rate, n_fft=1024, hop_length=512, cut_db=-10):
    window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(waveforms, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True, normalized=False, center=True)
    magnitude = stft.abs()
    db_magnitude = 20 * torch.log10(magnitude + 1e-6)
    db_magnitude[db_magnitude < cut_db] = -10

    # Create frequency vector in Hz (from 0 to Nyquist frequency)
    freqs = torch.linspace(0, sample_rate / 2, steps=n_fft // 2 + 1, device=device)

    return stft, db_magnitude, freqs



def plot_spectrogram(db_magnitude, freqs, sample_rate, hop_length, max_freq=5000):
    # Convert to CPU if needed
    db_magnitude = db_magnitude.squeeze(0).cpu().numpy()  # Remove batch dim if present
    freqs = freqs.cpu().numpy()

    # Mask frequencies above max_freq
    valid_indices = freqs <= max_freq
    db_magnitude = db_magnitude[valid_indices, :]
    freqs = freqs[valid_indices]

    # Time axis in seconds
    time_bins = db_magnitude.shape[1]
    times = torch.arange(time_bins) * (hop_length / sample_rate)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(db_magnitude, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()],
               cmap='magma')
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Spectrogram (Frequencies ≤ {max_freq} Hz)")
    plt.show()

# def display_stft(spec):
    # spec = spec.cpu().numpy()
    # plt.figure(figsize=(10, 4))
    # plt.imshow(spec, origin='lower', aspect='auto', interpolation='nearest')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectrogram')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.show()

def time_plot(signal, sample_rate=44100, show=True):
    if isnotebook():
        t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
        plt.plot(t, signal)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        if show:
            plt.show()

def stft_plot(signal, sample_rate=44100):
    if isnotebook():  # pragma: no cover
        X = librosa.stft(signal)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
        plt.show()
        
def write_wav(audio, sample_rate, file_name):
    if audio.device == "cuda":
        audio = audio.cpu()
    sf.write(file_name, audio, sample_rate)

def write_wavs(audio, sample_rate, folder, file_name, params_to_add=None):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    params_strings = []
    if params_to_add is not None:
        for i in range(len(audio)):
            current_params = []
            for name, values in params_to_add.items():
                current_params.append(name + "=" + str(round(float(values[i].item()), 2)))
            params = "-".join(current_params)
            params_strings.append(params)
    else:
        params_strings = len(audio) * [""]
    if not os.path.exists(folder):
        os.makedirs(folder)
    if "cuda" in str(audio.device):
        audio = audio.cpu()
    for i, ad in enumerate(audio):
        sf.write(folder + "/" + file_name + str(i) + "-" + params_strings[i] + ".wav", ad, sample_rate)
    print(str(audio.shape[0]) + " audio files written to " + folder)

