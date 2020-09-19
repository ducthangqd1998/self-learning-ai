import os
import cv2
import json
import shutil
import random
from glob import glob
from os.path import join
from tqdm import tqdm
from PIL import Image

def write_json(json_file, arr):
    with open(json_file, 'w') as file:
        json.dump(arr, file, indent=4)

def check_bad_image(path):
    u = 0
    with tqdm(total = len(path)) as epoch_pbar:
        for image in path:
            desc = f'Outfit { u + 1}'
            epoch_pbar.set_description(desc)
            epoch_pbar.update(1)
            u += 1
            try:
                img = Image.open(image)
                img.verify()
            except (IOError, SyntaxError) as e:
                os.remove(image)
                print('Bad file:', image)



train = glob('dataset/Original/Training/*')
test = glob('dataset/Original/Testing/*')

check_bad_image(train)
check_bad_image(test)

random.shuffle(train)
random.shuffle(test)
dataset = dict()
train_ = list()
test_ = list()

for i in train:
    train_.append(i.split('/')[-1])

for i in test:
    test_.append(i.split('/')[-1])

print(len(train_))
dataset['train'] = train_
dataset['test'] = test_

write_json('dataset/data.json', dataset)



