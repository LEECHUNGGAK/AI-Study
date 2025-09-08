# dp_matching_knn.py


import numpy as np

from dp_matching.dp_matching import dp_matching


if __name__ == "__main__":
    num_set = 5
    num_utt = 10
    query_set = 1
    query_utt = 9
    K = 3
    num_dims = 13

    query_file = f"./mfcc/REPEAT500_set{query_set}_{query_utt:03d}.bin"
    query = np.fromfile(query_file, dtype=np.float32)
    query = query.reshape(-1, num_dims)

    cost = []
    for set_idx in range(2, num_set + 1):
        for utt_idx in range(1, num_utt + 1):
            target_file = f"./mfcc/REPEAT500_set{set_idx}_{utt_idx:03d}.bin"
            target = np.fromfile(target_file, dtype=np.float32)
            target = target.reshape(-1, num_dims)

            tmp_cost, tmp_path = dp_matching(query, target)
            cost.append(
                {
                    "utt": utt_idx,
                    "set": set_idx,
                    "cost": tmp_cost,
                }
            )

    cost = sorted(cost, key=lambda x: x["cost"])
    for i in range(len(cost)):
        print(
            f"{i + 1}: "
            f"utt: {cost[i]['utt']}"
            f"set: {cost[i]['set']}"
            f"cost: {cost[i]['cost']}"
        )

    voting = np.zeros(num_utt, np.int16)
    # 비용이 작은 K개 발화 중에 빈도가 가장 높은 UTT를 출력합니다.
    for i in range(K):
        voting[cost[i]["utt"] - 1] += 1

    max_voted = np.argmax(voting) + 1
    print("인식된 발화 ID:", max_voted)
