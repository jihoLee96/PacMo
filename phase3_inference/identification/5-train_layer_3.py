# NOTES_TRAIN = 150
NOTES_TRAIN = 150
NOTES_VALIDATE = 5
NUM_TREES = 200
NUM_USERS = 500

print("Importing Libraries...")
import time
import torch
import numpy as np
from lightgbm import LGBMClassifier, log_evaluation
from joblib import dump
import os

# ✅ 저장 폴더 미리 생성
os.makedirs("./models/layer3", exist_ok=True)
os.makedirs("./stats/training/layer3", exist_ok=True)

print("Importing Data...")
trainData = torch.load('./data/train.pt')
validateData = torch.load('./data/validate.pt')
groups = open("./data/groups.txt").read().split("\n")
groups = [list(map(int, group.split(","))) for group in groups]

def getClassifyData(data):
    dataX = data[:, 1:]
    dataY = data[:, 0]
    return dataX, dataY

for round in range(len(groups)):
    group = groups[round]
    print("Starting Round " + str(round+1) + "/" + str(len(groups)) + "...")

    print("Selecting Data...")
    trainFrame = []
    validateFrame = []
    for idx in group:
        # trainFrame.append(trainData[150*idx:NOTES_TRAIN+150*idx])
        # validateFrame.append(validateData[5*idx:NOTES_VALIDATE+5*idx])
        trainFrame.append(trainData[NOTES_TRAIN*idx:NOTES_TRAIN+NOTES_TRAIN*idx])
        validateFrame.append(validateData[NOTES_VALIDATE*idx:NOTES_VALIDATE+NOTES_VALIDATE*idx])


    print("Processing Data...")
    trainX, trainY = getClassifyData(torch.cat(trainFrame))
    validateX, validateY = getClassifyData(torch.cat(validateFrame))

    # ✅ unseen label 제거 (안정성용)
    validate_mask = torch.isin(validateY, trainY.unique())
    if not validate_mask.all():
        removed = torch.unique(validateY[~validate_mask])
        print(f"[INFO] Removed unseen labels: {removed.tolist()}")
        validateX = validateX[validate_mask]
        validateY = validateY[validate_mask]
        
    print("Training Model " + str(round+1) + "/" + str(len(groups)) + "...")
    clf = LGBMClassifier(boosting_type='goss', colsample_bytree=0.6933333333333332, learning_rate=0.1, \
        max_bin=63, max_depth=-1, min_child_weight=7, min_data_in_leaf=20, \
        min_split_gain=0.9473684210526315, n_estimators=NUM_TREES, \
        num_leaves=33, reg_alpha=0.7894736842105263, reg_lambda=0.894736842105263, \
        subsample=1, n_jobs=16, objective='multiclassova', device_type='gpu')
    start_time = time.time()
    clf.fit(trainX, trainY.long(),
           eval_set=[(validateX, validateY)],
           eval_metric='multi_error',
           callbacks=[log_evaluation()])
    end_time = time.time()
    print("Training Finished in %s Minutes" % ((end_time - start_time) / 60))

    print("Saving Model " + str(round+1) + "/" + str(len(groups)) + "...")
    dump(clf, './models/layer3/model' + str(round) + '.pkl')
    file = open("./stats/training/layer3/" + str(round) + ".txt", "w")
    file.write(str(end_time - start_time))
    file.close()
