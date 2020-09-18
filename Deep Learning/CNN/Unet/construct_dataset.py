import os
import shutil
from os.path import join, exists
from PIL import Image
from glob import glob
from tqdm import tqdm


def pbm2png():
    dir_path = 'dataset'

    if not os.path.exists(join(dir_path, 'GT')):
        raise Exception("There is no pbm image here."
                        "May be the image has been converted to png")

    os.makedirs(join(dir_path, 'MASKS'), exist_ok=True)
    os.makedirs(join(dir_path, 'MASKS', 'Training'), exist_ok=True)
    os.makedirs(join(dir_path, 'MASKS', 'Testing'), exist_ok=True)

    img_paths = glob(join(dir_path, 'GT', '*', '*'))

    for img_path in tqdm(img_paths):
        img = Image.open(img_path)
        name = img_path.split('/')[-1].split('.')[0]
        phase = img_path.split('/')[-2]

        img.save(join(dir_path, 'MASKS', phase, name + '.png'))

    shutil.rmtree(join(dir_path, 'GT'))




def main():
    pass

if __name__ == '__main__':
    main()


