import wave
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    wav_file = "../data/wav/BASIC5000_0001.wav"
    # 분석할 wav 파일에서 해당 시간에 음소 "o"를 발화합니다.
    target_time = 0.58
    # FFT(고속 푸리에 변환)을 적용할 샘플 범위
    fft_size = 1024
    out_plot = "./spectrum.png"
    
    with wave.open(wav_file) as wav:
        sampling_frequency = wav.getframerate()
        num_samples = wav.getnframes()
        waveform = wav.readframes(num_samples)
        waveform = np.frombuffer(waveform, dtype=np.int16)
    
    # 분석 시간을 샘플링 인덱스로 변환합니다.
    target_index = np.int32(target_time * sampling_frequency)
    frame = waveform[target_index : target_index + fft_size]
    spectrum = np.fft.fft(frame)
    # 진폭 스펙트럼
    absolute = np.abs(spectrum)
    # 진폭 스펙트럼은 분석 구간에서 좌우 대칭이므로 좌측만 사용합니다.
    half_size = np.int32(fft_size / 2) + 1
    absolute = absolute[:half_size]
    # 로그 함수를 취합니다.
    log_absolute = np.log(absolute + 1e-7)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    time_axis = np.arange(num_samples) / sampling_frequency
    plt.plot(time_axis, waveform)
    plt.title("waveform")
    plt.xlabel("Time [sec]")
    plt.ylabel("Value")
    plt.xlim([0, num_samples / sampling_frequency])
    
    plt.subplot(2, 1, 2)
    # x축 : 절반의 진폭 스펙트럼 * FFT 변환된 스펙트럼의 개수
    freq_axis = np.arange(half_size) * sampling_frequency / fft_size
    plt.plot(freq_axis, log_absolute)
    plt.title("log-absolute spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Value")
    plt.xlim([0, sampling_frequency / 2])
    
    plt.savefig(out_plot)

    plt.figure(figsize=(10, 10))
    plt.plot(freq_axis, absolute)
    plt.title("absolute spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Value")
    plt.xlim([0, sampling_frequency / 2])
    plt.savefig("./raw_spectrum.png")