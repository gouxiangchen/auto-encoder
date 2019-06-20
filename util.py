import numpy as np
import os
import tqdm
import cv2


def load_mnist_data(path):
    fd = open(os.path.join(path, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 784)).astype(np.float)

    fd = open(os.path.join(path, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)
    return trX, trY


def get_file_name(dir):
    filenames = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


def load_grasp_img(path):
    k = 0
    imgs = np.zeros((1, 224, 224)).astype(np.uint8)
    filenames = get_file_name(path)
    for filename in tqdm.tqdm(filenames):
        img = cv2.imread(filename, 0)
        img = np.asarray(img).astype(np.uint8)
        if k == 0:
            imgs[0] = img
            k += 1
        else:
            img = img.reshape((1, 224, 224))
            imgs = np.append(imgs, img, axis=0)
    return imgs
