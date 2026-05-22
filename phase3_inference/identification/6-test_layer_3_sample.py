NOTES_TEST = 50
NUM_TREES = 200
NUM_USERS = 500

print("Importing Libraries...")
import time
import torch
import numpy as np
from lightgbm import LGBMClassifier, log_evaluation
from joblib import load
from tqdm import tqdm
import os

os.makedirs("./preds/test/layer3", exist_ok=True)
os.makedirs("./stats/testing/layer3", exist_ok=True)

print("Importing Data...")
testData = torch.load('./data/test.pt')
groups = open("./data/groups.txt").read().split("\n")
groups = [list(map(int, group.split(","))) for group in groups]

print("Processing Data...")
def getClassifyData(data):
    dataX = data[:, 1:]
    dataY = data[:, 0]
    return dataX, dataY

testX, testY = getClassifyData(testData)

print("Importing Models...")
clfs3 = []
for i in tqdm(range(len(groups))):
    clfs3.append(load(f'./models/layer3/model{i}.pkl'))

for round in range(len(groups)):
    group = groups[round]
    print(f"\n=== Starting Round {round+1}/{len(groups)} ===")

    start_time = time.time()
    mtrxCtest = []
    valid_user = 0
    valid_sample = 0
    total_user = 0
    total_sample = 0

    t = tqdm(group, desc='0/0 Valid (0%)')

    for i in t:
        # (1) 전체 50개 샘플의 확률 예측
        preds3 = clfs3[round].predict_proba(testX[NOTES_TEST*i:NOTES_TEST+NOTES_TEST*i])
        pred_sum = preds3.sum(axis=0)
        mtrxCtest.append(pred_sum)

        # (2) per-user accuracy
        if clfs3[round].classes_[np.argmax(pred_sum)] == i:
            valid_user += 1
        total_user += 1

        # (3) per-sample accuracy
        sample_preds = np.argmax(preds3, axis=1)
        correct_samples = np.sum(clfs3[round].classes_[sample_preds] == i)
        valid_sample += correct_samples
        total_sample += len(sample_preds)

        t.set_description(
            f"User {valid_user}/{total_user} ({valid_user/total_user*100:.2f}%) | "
            f"Sample {valid_sample}/{total_sample} ({valid_sample/total_sample*100:.2f}%)"
        )

    end_time = time.time()
    print(f"Finished in {(end_time - start_time)/60:.2f} Minutes")

    # ✅ 결과 저장
    np.save(f'./preds/test/layer3/{round}', mtrxCtest)
    with open(f"./stats/testing/layer3/{round}.txt", "w") as f:
        f.write(f"Time: {end_time - start_time:.2f}s\n")
        f.write(f"Per-user accuracy: {valid_user/total_user:.4f}\n")
        f.write(f"Per-sample accuracy: {valid_sample/total_sample:.4f}\n")

    print(f"[Round {round+1}] Per-user acc = {valid_user/total_user:.4f}, "
          f"Per-sample acc = {valid_sample/total_sample:.4f}")
