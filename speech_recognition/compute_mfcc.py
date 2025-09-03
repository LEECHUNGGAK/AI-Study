# compute_mfcc.py


import os
import wave
import numpy as np

from compute_fbank import FeatureExtractor as FeatureExtractorBase


class FeatureExtractor(FeatureExtractorBase):
    def __init__(
        self,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
        num_mel_bins=23,
        num_ceps=13,
        lifter_coef=22,
        low_frequency=20,
        high_frequency=8000,
        dither_coef=1.0,
    ):
        super().__init__(
            sample_frequency=sample_frequency,
            frame_length=frame_length,
            frame_shift=frame_shift,
            num_mel_bins=num_mel_bins,
            num_ceps=num_ceps,
            lifter_coef=lifter_coef,
            low_frequency=low_frequency,
            high_frequency=high_frequency,
            dither_coef=dither_coef,
        )
        self.dct_matrix = self.make_dct_matrix()
        self.lifter = self.make_lifter()
    
    def make_dct_matrix(self):
        N = self.num_mel_bins
        dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2 / N) * np.cos((2 * np.arange(N) + 1) * k * np.pi / (2 * N))

        return dct_matrix
    
    def make_lifter(self):
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1 + Q / 2 * np.sin(np.pi * I / Q)
        return lifter
    
    def compute_mfcc(self, waveform):
        fbank, log_power = self.compute_fbank(waveform)
        mfcc = np.dot(fbank, self.dct_matrix.T)
        
        # 리프터링
        mfcc *= self.lifter
        
        # MFCC 0차원에 로그 파워를 할당합니다.
        mfcc[:, 0] = log_power
        
        return mfcc


if __name__ == "__main__":
    train_small_wav_scp = "../data/label/train_small/wav.scp"
    train_small_out_dir = "./mfcc/train_small"
    train_large_wav_scp = "../data/label/train_large/wav.scp"
    train_large_out_dir = "./mfcc/train_large"
    dev_wav_scp = "../data/label/dev/wav.scp"
    dev_out_dir = "./mfcc/dev"
    test_wav_scp = "../data/label/test/wav.scp"
    test_out_dir = "./mfcc/test"
    
    sample_frequency = 16000
    frame_length = 25
    frame_shift = 10
    low_frequency = 20
    high_frequency = sample_frequency / 2
    num_mel_bins = 23
    num_ceps = 13
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
        os.makedirs(out_dir, exist_ok=True)
        feat_scp = os.path.join(out_dir, "feats.scp")
        
        with open(wav_scp, mode="r") as file_wav, open(feat_scp, mode="w") as file_feat:
            for line in file_wav:
                parts = line.split()
                utterance_id, wav_path = parts

                with wave.open(wav_path) as wav:
                    num_samples = wav.getnframes()
                    waveform = wav.readframes(num_samples)
                    waveform = np.frombuffer(waveform, dtype=np.int16)
                    mfcc = feature_extractor.compute_mfcc(waveform)
                
                (num_frames, num_dims) = np.shape(mfcc)
                out_file = os.path.splitext(os.path.basename(wav_path))[0]
                out_file = os.path.join(os.path.abspath(out_dir), out_file + ".bin")

                mfcc = mfcc.astype(np.float32)
                mfcc.tofile(out_file)
                
                file_feat.write(f"{utterance_id} {out_file} {num_frames} {num_dims}\n")