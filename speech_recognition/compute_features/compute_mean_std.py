# compute_mean_std.py


import os
import wave
import numpy as np


if __name__ == "__main__":
    feature_list = ["fbank", "mfcc"]
    for feature in feature_list:
        feat_scp = f"./{feature}/train_small/feats.scp"
        out_dir = f"./{feature}/train_small"
        
        feat_mean = None
        feat_var = None
        total_frames = 0
        
        with open(feat_scp, mode="r") as file_feat:
            for i, line in enumerate(file_feat):
                parts = line.split()
                utterance_id = parts[0]
                feat_path = parts[1]
                num_frames = int(parts[2])
                num_dims = int(parts[3])

                # 저장할 때 1행 벡터 크기로 저장된 특징값을 (프레임 수, 차원 수) 크기로 변환합니다.
                feature = np.fromfile(feat_path, dtype=np.float32)
                feature = feature.reshape(num_frames, num_dims)
                
                # 최초에 평균과 분산을 초기화합니다.
                if i == 0:
                    feat_mean = np.zeros(num_dims, np.float32)
                    feat_var = np.zeros(num_dims, np.float32)
                
                # 합 계산
                feat_mean += np.sum(feature, axis = 0)
                # 제곱합 계산
                feat_var += np.sum(np.power(feature, 2))
                # 총 프레임 수 계산
                total_frames += num_frames
                
            feat_mean /= total_frames
            # 분산 = 제곱합의 평균 - 평균의 제곱
            feat_var = (feat_var / total_frames) - np.power(feat_mean, 2) 
            feat_std = np.sqrt(feat_var)
            
            out_file = os.path.join(out_dir, "mean_std.txt")
            print(f"출력 파일: {out_file}")
            with open(out_file, mode="w") as file_output:
                file_output.write("평균\n")
                for i in range(np.size(feat_mean)):
                    file_output.write(f"{feat_mean[i]} ")
                file_output.write("\n")
                
                file_output.write("표준편차\n")
                for i in range(np.size(feat_std)):
                    file_output.write(f"{feat_std[i]} ")
                file_output.write("\n")