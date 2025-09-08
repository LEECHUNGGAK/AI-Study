import wave
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    wav_file = "../data/wav/BASIC5000_0001.wav"
    # 프레임 사이즈와 프레임 시프트는 ms 단위를 사용합니다.
    frame_size = 25
    frame_shift = 10
    out_plot = "./spectrogram.png"

    with wave.open(wav_file) as wav:
        sample_frequency = wav.getframerate()
        num_samples = wav.getnframes()
        waveform = wav.readframes(num_samples)
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # 프레임 사이즈와 프레임 시프트의 단위를 샘플 수로 변환합니다.
    frame_size = int(frame_size * sample_frequency * 0.001)
    frame_shift = int(frame_shift * sample_frequency * 0.001)

    # 분석 구간의 크기를 프레임 사이즈보다 큰 2제곱으로 설정합니다.
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    num_frames = (num_samples - frame_size) // frame_shift + 1
    half_size = int(fft_size / 2) + 1

    # 시간의 변화에 따른 스펙트럼을 저장할 스펙트로그램 행렬
    spectrogram = np.zeros((num_frames, half_size))
    for idx in range(num_frames):
        start_idx = idx * frame_shift
        # 프레임 추출
        frame = waveform[start_idx : start_idx + frame_size].copy()
        # 프레임에 해밍 창 함수 적용
        frame = frame * np.hamming(frame_size)
        # 로그 진폭 스펙트럼 계산
        spectrum = np.fft.fft(frame, n=fft_size)
        absolute = np.abs(spectrum)
        absolute = absolute[:half_size]
        log_absolute = np.log(absolute + 1e-7)
        spectrogram[idx, :] = log_absolute

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    time_axis = np.arange(num_samples) / sample_frequency
    plt.plot(time_axis, waveform)
    plt.title("waveform")
    plt.xlabel("Time [sec]")
    plt.ylabel("Value")
    plt.xlim([0, num_samples / sample_frequency])

    # 주파수에 최소 최대 정규화를 적용합니다.
    spectrogram -= np.max(spectrogram)
    vmax = 0
    vmin = -np.abs(np.min(spectrogram)) * 0.7

    plt.subplot(2, 1, 2)
    plt.imshow(
        spectrogram.T,
        origin="lower",
        # 이미지의 [좌, 우, 하, 상] 범위
        extent=[
            0,
            num_samples / sample_frequency,
            0,
            sample_frequency / 2,
        ],
        cmap="gray",
        vmax=vmax,
        vmin=vmin,
        aspect="auto",
    )
    plt.title("spectrogram")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig(out_plot)
