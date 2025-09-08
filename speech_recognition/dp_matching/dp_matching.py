# dp_matching.py


import wave
import numpy as np
from pathlib import Path

from prepare.prepare_wav import WavPreprocessor
from compute_features.compute_mfcc import FeatureExtractor


def dp_matching(feature_1, feature_2):
    nframes_1, num_dims = np.shape(feature_1)
    nframes_2 = np.shape(feature_2)[0]

    # 비용 행렬, 누적 비용 행렬, 이동 종류를 초기화합니다.
    distance = np.zeros((nframes_1, nframes_2))
    for n in range(nframes_1):
        for m in range(nframes_2):
            # 유클리드 거리를 계산할 때 제곱근 계산은 생략한 유클리드 거리 제곱을 사용합니다.
            distance[n, m] = np.sum((feature_1[n] - feature_2[m]) ** 2)

    cost = np.zeros((nframes_1, nframes_2))
    cost[0, 0] = distance[0, 0]
    track = np.zeros((nframes_1, nframes_2), np.int16)
    
    # 무조건 세로로 이동하는 경우
    for n in range(1, nframes_1):
        cost[n, 0] = cost[n - 1, 0] + distance[n, 0]
        track[n, 0] = 0

    # 무조건 가로로 이동하는 경우
    for m in range(1, nframes_2):
        cost[0, m] = cost[0, m - 1] + distance[0, m]
        track[0, m] = 2

    # 최소 비용을 향해 이동하는 경우
    for n in range(1, nframes_1):
        for m in range(1, nframes_2):
            # 세로로 이동했을 때 누적 비용
            vertical = cost[n - 1, m] + distance[n, m]
            # 사선으로 이동했을 때 누적 비용
            diagonal = cost[n - 1, m - 1] + 2 * distance[n, m]
            # 가로로 이동했을 때 누적 비용
            horizontal = cost[n, m - 1] + distance[n, m]

            # 누적 비용이 최소인 경로를 선택합니다.
            candidate = [vertical, diagonal, horizontal]
            transition = np.argmin(candidate)
            cost[n, m] = candidate[transition]
            track[n, m] = transition

    # 총 비용을 정규화합니다.
    total_cost = cost[-1, -1] / (nframes_1 + nframes_2)

    # Back Track
    min_path = []
    n = nframes_1 - 1
    m = nframes_2 - 1
    while True:
        min_path.append([n, m])
        if n == 0 and m == 0:
            break

        if track[n, m] == 0:
            n -= 1
        elif track[n, m] == 1:
            n -= 1
            m -= 1
        else:
            m -= 1

    # 역방향의 최단 경로를 순방향으로 정렬합니다.
    min_path = min_path[::-1]

    return total_cost, min_path


if __name__ == "__main__":
    # Prepare wav
    original_wav_dir = Path("../data/original/jsut_ver1.1/repeat500/wav")
    out_wav_dir = Path("./wav")
    out_wav_dir.mkdir(exist_ok=True)
    num_set = 5
    num_utt_per_set = 10
    wav_preprocessor = WavPreprocessor()
    for set_id in range(num_set):
        for utt_id in range(num_utt_per_set):
            filename = f"REPEAT500_set{set_id+1}_{utt_id+1:03d}"
            wav_path_in = original_wav_dir / (filename + ".wav")
            wav_path_out = out_wav_dir / (filename + ".wav")
            if wav_path_out.exists:
                continue
            wav_preprocessor.downsample(
                input_filepath=wav_path_in,
                output_filepath=wav_path_out,
            )

    # Compute MFCC
    out_mfcc_dir = Path("./mfcc")
    np.random.seed(seed=0)
    feature_extractor = FeatureExtractor()
    for wav_path in sorted(out_wav_dir.iterdir()):
        mfcc_path_out = out_mfcc_dir / (wav_path.stem + ".bin")
        if mfcc_path_out.exists():
            continue

        with wave.open(str(wav_path)) as wav:
            num_samples = wav.getnframes()
            waveform = wav.readframes(num_samples)
            waveform = np.frombuffer(waveform, dtype=np.int16)
            mfcc = feature_extractor.compute_mfcc(waveform)
            mfcc = mfcc.astype(np.float32)
            mfcc.tofile(mfcc_path_out)

    # DP matching
    mfcc_file_1 = "./mfcc/REPEAT500_set1_009.bin"
    mfcc_file_2 = "./mfcc/REPEAT500_set2_009.bin"
    result = "./alignment.txt"
    # MFCC 차원 수
    num_dims = 13
    mfcc_1 = np.fromfile(mfcc_file_1, dtype=np.float32)
    mfcc_2 = np.fromfile(mfcc_file_2, dtype=np.float32)
    mfcc_1 = mfcc_1.reshape(-1, num_dims)
    mfcc_2 = mfcc_2.reshape(-1, num_dims)

    total_cost, min_path = dp_matching(mfcc_1, mfcc_2)
    with open(result, mode="w") as f:
        for p in min_path:
            f.write(f"{p[0]} {p[1]}\n")
