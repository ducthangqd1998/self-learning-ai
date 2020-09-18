import os
import cv2
import json
import shutil
import random
from glob import glob
from os.path import join


def write_json(json_file, arr):
    with open(json_file, 'w') as file:
        json.dump(arr, file, indent=4)


train = glob('dataset/Original/Training/*')
test = glob('dataset/MASKS/Testing/*')

random.shuffle(train)
random.shuffle(test)
dataset = dict()
train_ = list()
test_ = list()

for i in train:
    train_.append(i.split('/')[-1])

for i in test_:
    test_.append(i.split('/')[-1])

print(len(train_))
dataset['train'] = train_
dataset['test'] = test_

write_json('dataset/data.json', dataset)



