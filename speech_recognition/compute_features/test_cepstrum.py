# test_cepstrum.py


import os
import wave
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    wav_file = "../data/wav/BASIC5000_0001.wav"
    # 분석 시간. 해당 시간에 음소 "오"를 발화합니다.
    target_time = 0.58
    fft_size = 1024
    # 켑스트럼에서 저차 영역과 고차 영역의 경계
    cep_threshold = 33
    out_plot = "./cepstrum.png"
    
    with wave.open(wav_file) as wav:
        sampling_frequency = wav.getframerate()
        waveform = wav.readframes(wav.getnframes())
        waveform = np.frombuffer(waveform, dtype=np.int16)
    
    target_index = np.int16(target_time * sampling_frequency)
    frame = waveform[target_index : target_index + fft_size].copy()
    
    # 해밍 창 함수 적용
    frame = frame * np.hamming(fft_size)
    
    # 로그 파워 스펙트럼 계산
    spectrum = np.fft.fft(frame)
    log_power = 2 * np.log(np.abs(spectrum) + 1e-7)
    
    # 로그 파워 스펙트럼에 역푸리에 변환을 적용해 켑스트럼을 구합니다.
    cepstrum = np.fft.ifft(log_power)
    
    # 고차 영역을 제거합니다.
    cepstrum_low = cepstrum.copy()
    cepstrum_low[cep_threshold + 1 : -cep_threshold] = 0.0
    
    # 고차 영역을 제거한 켑스트럼을 푸리에 변환하여,
    # 성도 공진 특성에 대한 로그 파워 스펙트럼을 구합니다.
    log_power_ceplo = np.abs(np.fft.fft(cepstrum_low))
    
    # 고차 영역만 추출합니다.
    cepstrum_high = cepstrum - cepstrum_low
    # 저차 영역의 기본 주파수는 추가합니다.
    cepstrum_high[0] = cepstrum[0]
    
    # 고차 영역을 푸리에 변환하여 로그 파워 스펙트럼을 구합니다.
    log_power_cephi = np.fft.fft(cepstrum_high)
    
    plt.figure(figsize=(18, 10))
    # 로그 파워 스펙트럼을 시각화하기 위해 x축(주파수 축)을 생성합니다.
    freq_axis = np.arange(fft_size) * sampling_frequency / fft_size

    for idx, elem in enumerate([log_power, log_power_ceplo, log_power_cephi]):
        plt.subplot(3, 2, idx * 2 + 1)
        plt.plot(freq_axis, elem, color="k")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Value")
        plt.xlim([0, sampling_frequency / 2])
        plt.ylim([0, 30])
    
    # 켑스트럼의 x축(큐프렌시 축)을 생성합니다.
    qefr_axis = np.arange(fft_size) / sampling_frequency
    
    for idx, elem in enumerate([cepstrum, cepstrum_low, cepstrum_high]):
        plt.subplot(3, 2, idx  * 2 + 2)
        # 캡스트럼의 실수부만 시각화합니다.
        plt.plot(qefr_axis, np.real(elem), color="k")
        plt.xlabel("QueFrency [sec]")
        plt.ylabel("Value")
        plt.xlim([0, fft_size / (sampling_frequency * 2)])
        plt.ylim([-1, 2])

    plt.savefig(out_plot)