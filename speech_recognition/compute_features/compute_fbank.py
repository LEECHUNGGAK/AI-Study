# compute_fbank.py


import os
import wave
import numpy as np


class FeatureExtractor:
    def __init__(
        self,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
        num_mel_bins=23,
        num_ceps=13,
        lifter_coef=22,
        # 분석 시작 주파수
        low_frequency=20,
        # 분석 종료 주파수
        high_frequency=8000,
        dither_coef=1.0,
    ):
        """
        num_mel_bins: Mel 필터 뱅크 수
        num_ceps: MFCC 특징 차원 수
        lifter_coef: 리프터링 처리 매개 변수
        low_frequency: 분석 시작 주파수
        high_frequency: 분석 종료 주파수
        """

        self.sample_frequency = sample_frequency
        # 창의 단위를 밀리초에서 샘플 수로 변환합니다.
        self.frame_length = int(sample_frequency * 0.001 * frame_length)
        self.frame_shift = int(sample_frequency * 0.001 * frame_shift)
        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.lifter_coef = lifter_coef
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.dither_coef = dither_coef
        
        self.fft_size = 1
        while self.fft_size < self.frame_length:
            self.fft_size *= 2
        
        self.mel_filter_bank = self.make_mel_filter_bank()
    
    def herz2mel(self, herz):
        """주파수를 Mel 척도로 변환합니다."""
        
        return 1127 * np.log(1 + herz / 700)
    
    def make_mel_filter_bank(self):
        mel_high_freq = self.herz2mel(self.high_frequency)
        mel_low_freq = self.herz2mel(self.low_frequency)
        mel_points = np.linspace(
            mel_low_freq,
            mel_high_freq,
            # 시작점, 끝점을 포함시키기 위해 2를 더합니다.
            self.num_mel_bins + 2,
        )
        # 스펙트럼의 차원 수
        dim_spectrum = int(self.fft_size / 2) + 1
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        
        # m : 필터 기준
        for m in range(self.num_mel_bins):
            # 삼각 필터 좌측, 중앙, 우측
            left_mel = mel_points[m]
            center_mel = mel_points[m + 1]
            right_mel = mel_points[m + 2]
            
            # n : 각각의 스펙트럼에 따른 필터값 계산
            for n in range(dim_spectrum):
                freq = n * self.sample_frequency / 2 / dim_spectrum
                mel = self.herz2mel(freq)
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    mel_filter_bank[m][n] = weight

        return mel_filter_bank

    def extract_window(
        self,
        waveform,
        start_index,
    ):
        """
        파형 데이터에서 1프레임을 추출하여 전처리합니다.
        """
        window = waveform[start_index : start_index + self.frame_length].copy()
        
        # 디더링
        if self.dither_coef > 0:
            window = window + np.random.rand(self.frame_length) * (2 * self.dither_coef) - self.dither_coef
        
        # 직류 성분 제거
        window = window - np.mean(window)
        
        # 파워 계산
        power = np.sum(window ** 2)
        
        # 로그를 취할 때 -inf가 되지 않도록 플로어링 처리
        if power < 1e-10:
            power = 1e-10
        
        # 로그 적용
        log_power = np.log(power)
        
        # 고역 강조
        window = np.convolve(
            window,
            np.array([1.0, -0.97]),
            mode="same",
        )
        """
        numpy.convolve(a, v)는 두 개의 일차원 시퀀스의 이산 선형 합성곱을 구합니다.
        
        mode="full" : 출력의 차원 (N+M-1, )
        a=[1, 2, 3], v=[0, 1, 0.5] 일 때
        출력 크기 = M(a의 길이) + N(b의 길이) - 1
        고정된 a에 b를 이동시키며 연산
        k = 0 : a[0] * v[0] = 1 * 0
        k = 1 : v를 한 칸 이동 a[1] * v[0] + a[0] * v[1] = 2 * 0 + 1 * 1
        k = 2 : v를 한 칸 이동 a[2] * v[0] + a[1] * v[1] + a[2] * v[0]= 3 * 0 + 2 * 1 + 1 * 0.5
        k = 3 : a[2] * v[1] + a[1] * v[2]= 3 * 1 + 2 * 0.5
        k = 4 : a[2] * v[2]= 3 * 0.5
        
        mode="same" : 출력의 차원 max(M, N)
        a의 영역에서 v의 중앙에 위치하도록 이동하며 연산합니다.
        k = 0 : a[1] * v[0] + a[0] * v[1]
        k = 1 : a[2] * v[0] + a[1] * v[1] + a[0] * v[2]
        k = 2 : a[2] * v[1] + a[1] * v[2]
        
        그런데 첫 번째 연산, window[0] = 1 * window[0] - 0.97 * window[0 - 1],에서
        window[0] 이전의 값은 없으므로 window[0 - 1]은 window[0]으로 대체하여 연산합니다.
        이 연산은 수동으로 처리합니다.
        """
        window[0] -= 0.97 * window[0]
        
        # 해밍 창 함수 적용
        window *= np.hamming(self.frame_length)
        
        return window, log_power
    
    def compute_fbank(self, waveform):
        num_samples = np.size(waveform)
        num_frames = (num_samples - self.frame_length) // self.frame_shift + 1

        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        log_powers = np.zeros(num_frames)
        half_size = int(self.fft_size / 2) + 1

        for frame in range(num_frames):
            start_index = frame * self.frame_shift
            window, log_power = self.extract_window(waveform, start_index)
            
            # 파워 스펙트럼 계산
            spectrum = np.fft.fft(window, n=self.fft_size)
            spectrum = spectrum[:half_size]
            spectrum = np.abs(spectrum) ** 2
            
            # Mel 필터 뱅크 계산
            fbank = np.dot(spectrum, self.mel_filter_bank.T)
            
            # 플로어링 처리
            fbank[fbank < 0.1] = 0.1
            
            # 로그 처리 후 결과값에 할당
            fbank_features[frame] = np.log(fbank)
            log_powers[frame] = log_power
        
        return fbank_features, log_powers


if __name__ == "__main__":
    train_small_wav_scp = "../data/label/train_small/wav.scp"
    train_small_out_dir = "./fbank/train_small"
    train_large_wav_scp = "../data/label/train_large/wav.scp"
    train_large_out_dir = "./fbank/train_large"
    dev_wav_scp = "../data/label/dev/wav.scp"
    dev_out_dir = "./fbank/dev"
    test_wav_scp = "../data/label/test/wav.scp"
    test_out_dir = "./fbank/test"
    
    sample_frequency = 16000
    frame_length = 25
    frame_shift = 10
    low_frequency = 20
    high_frequency = sample_frequency / 2
    num_mel_bins = 40
    dither_coef = 1
    
    feature_extractor = FeatureExtractor(
        sample_frequency=sample_frequency,
        frame_length=frame_length,
        frame_shift=frame_shift,
        num_mel_bins=num_mel_bins,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        dither_coef=dither_coef,
    )
    
    wav_scp_list = [
        train_small_wav_scp,
        train_large_wav_scp,
        dev_wav_scp,
        test_wav_scp,
    ]
    out_dir_list = [
        train_small_out_dir,
        train_large_out_dir,
        dev_out_dir,
        test_out_dir,
    ]
    
    for (wav_scp, out_dir) in zip(wav_scp_list, out_dir_list):
        print(f"입력 wav_scp: {wav_scp}")
        print(f"출력 디렉토리: {out_dir}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        feat_scp = os.path.join(out_dir, "feats.scp")
        
        with open(wav_scp, mode="r") as file_wav, open(feat_scp, mode="w") as file_feat:
            # scp 파일의 각 행에는 발화 ID, wav 파일 경로가 스페이스로 구분되어 있습니다.
            for line in file_wav:
                parts = line.split()
                utterance_id, wav_path = parts

                with wave.open(wav_path) as wav:
                    num_samples = wav.getnframes()
                    waveform = wav.readframes(num_samples)
                    waveform = np.frombuffer(waveform, dtype=np.int16)
                    fbank, log_power = feature_extractor.compute_fbank(waveform)
                
                (num_frames, num_dims) = np.shape(fbank)
                # 확장자를 제외한 이름 추출
                out_file = os.path.splitext(os.path.basename(wav_path))[0]
                out_file = os.path.join(os.path.abspath(out_dir), out_file + ".bin")
                
                fbank = fbank.astype(np.float32)
                fbank.tofile(out_file)
                
                file_feat.write(f"{utterance_id} {out_file} {num_frames} {num_dims}\n")