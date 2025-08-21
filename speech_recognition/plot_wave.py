import wave
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    wav_file = "../data/wav/BASIC5000_0001.wav"
    out_plot = "./plot.png"

    with wave.open(wav_file) as wav:
        sampling_frequency = wav.getframerate()
        sampling_size = wav.getsampwidth()
        channel_number = wav.getnchannels()
        sample_number = wav.getnframes()

        # 데이터 읽기
        waveform = wav.readframes(sample_number)
        # 음압치가 16비트 바이너리 자료형이므로 정수로 변환합니다.
        waveform = np.frombuffer(waveform, dtype=np.int16)

    print(f"샘플링 주파수: {sampling_frequency} [Hz]")
    print(f"샘플링 사이즈: {sampling_size} [Byte]")
    print(f"채널 수: {channel_number}")
    print(f"샘플 수: {sample_number}")

    # 그래프 영역 생성
    plt.figure(figsize=(10, 4))

    # 시간 축 생성
    time_axis = np.arange(sample_number) / sampling_frequency

    # 파형 생성
    plt.plot(time_axis, waveform)

    # 그래프 레이블 생성
    plt.xlabel("시간 [sec]")
    plt.ylabel("음압치")

    # 시간 축의 범위 설정
    plt.xlim([0, sample_number / sampling_frequency])

    # 그래프 저장
    plt.savefig(out_plot)
