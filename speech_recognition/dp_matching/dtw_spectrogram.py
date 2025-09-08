# dwt_spectrogram.sorted


import wave
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    wav_file_1 = "./wav/REPEAT500_set1_009.wav"
    wav_file_2 = "./wav/REPEAT500_set2_009.wav"
    alignment_file = "./alignment.txt"
    out_plot = "./dtw_spectrogram.png"
    sample_frequency = 16000
    frame_size = 25
    frame_shift = 10

    frame_size = int(frame_size * sample_frequency * 0.001)
    frame_shift = int(frame_shift * sample_frequency * 0.001)

    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    alignment = []
    with open(alignment_file, mode="r") as f:
        for line in f:
            parts = line.split()
            alignment.append([int(parts[0]), int(parts[1])])

    plt.figure(figsize=(20, 10))
    for wav_idx, wav_file in enumerate([wav_file_1, wav_file_2]):
        with wave.open(wav_file) as wav:
            num_samples = wav.getnframes()
            waveform = wav.readframes(num_samples)
            waveform = np.frombuffer(waveform, dtype=np.int16)

        num_frames = (num_samples - frame_size) // frame_shift + 1
        spectrogram = np.zeros((num_frames, fft_size))
        for frame_idx in range(num_frames):
            start_idx = frame_idx * frame_shift
            frame = waveform[start_idx : start_idx + frame_size].copy()
            frame = frame * np.hamming(frame_size)
            spectrum = np.fft.fft(frame, n=fft_size)
            log_spectrum = np.log(np.abs(spectrum) + 1e-7)
            spectrogram[frame_idx, :] = log_spectrum

        plt.subplot(2, 2, wav_idx * 2 + 1)
        spectrogram_norm = spectrogram - np.max(spectrogram)
        vmax = 0
        vmin = -np.abs(np.min(spectrogram_norm)) * 0.7
        plt.imshow(
            spectrogram_norm.T[-1::-1, :],
            extent=[
                0,
                num_samples / sample_frequency,
                0,
                sample_frequency,
            ],
            cmap="gray",
            vmax=vmax,
            vmin=vmin,
            aspect="auto",
        )
        plt.ylim([0, sample_frequency / 2])
        plt.title("spectrogram")
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [Hz]")

        dtw_spectrogram = np.zeros((len(alignment), fft_size))
        for t in range(len(alignment)):
            # t 시점에 최소 비용인 프레임의 위치
            idx = alignment[t][wav_idx]
            dtw_spectrogram[t, :] = spectrogram[idx, :]

        plt.subplot(2, 2, wav_idx * 2 + 2)
        dtw_spectrogram -= np.max(dtw_spectrogram)
        vmax = 0
        vmin = -np.abs(np.min(dtw_spectrogram)) * 0.7
        plt.imshow(
            dtw_spectrogram.T[-1::-1, :],
            extent=[
                0,
                len(alignment) * frame_shift / sample_frequency,
                0,
                sample_frequency,
            ],
            cmap="gray",
            vmax=vmax,
            vmin=vmin,
            aspect="auto",
        )
        plt.ylim([0, sample_frequency / 2])
        plt.title("spectrogram")
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [Hz]")

    plt.savefig(out_plot)
