import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile


sources = [
    "C:/Users/lundb/Documents/Other/Music/i.wav",
    # "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_0.7.wav", 
    # "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_1.wav",
    # "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_1.5.wav",
    ]

plt.figure(figsize=(12, 6))

for i, source in enumerate(sources):
    if ".npy" in source:
        s = np.load(source)[0].squeeze()
        t = np.arange(len(s))
        sr = 44100
    elif ".wav" in source:
        sr, s = wavfile.read(source)
        if s.shape[1]>1:
            s=s[:,0]
        t = np.arange(len(s))

    fft_result = np.fft.fft(s)
    freq_values = np.fft.fftfreq(len(fft_result), 1/sr)

    # Compute the spectrogram
    frequencies, times, spectrogram_data = spectrogram(s, fs=sr)

    
    plt.subplot(3, len(sources), 1+i)
    plt.plot(t, s)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal in Time Domain')
    plt.grid()

    plt.subplot(3, len(sources), (1+len(sources))+i)
    plt.plot(freq_values[:len(freq_values)//2], np.abs(fft_result)[:len(freq_values)//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Audio Signal in Frequency Domain')
    plt.grid()

    plt.subplot(3, len(sources), (1+len(sources)*2)+i)
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='auto')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.colorbar(label='dB')

plt.tight_layout()
plt.show()