import cv2
import tflearn
import os
import numpy as np
import pickle as pkl
from cnn_model import *


def load_data(datafile, num_classes, save=True, img_size=224, save_path="dataset.pkl"):
    with open(datafile, "r") as f:
        img_list = f.readlines()
    images_data = []
    labels = []
    for line in img_list:
        line = line.strip().split(" ")
        file_path = line[0]
        img = cv2.imread(file_path)
        img = cv2.resize(img, (img_size, img_size), cv2.INTER_CUBIC)
        img_np = np.array(img, dtype="float32")
        images_data.append(img_np)
        index = int(line[1])
        label = np.zeros(num_classes)
        label[index] = 1
        labels.append(label)
    if save:
        with open(save_path, "wb") as s:
            pkl.dump((images_data, labels), s)
    return images_data, labels


def load_data_from_pkl(file_path):
    with open(file_path, "rb") as f:
        images_data, labels = pkl.load(f)
    return images_data, labels


def train(network, x, y, save_model_path):
    model = tflearn.DNN(network, checkpoint_path="./check_point/model_alexnet",
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="output")
    if os.path.isfile(save_model_path + ".index"):
        model.load(save_model_path)
        print("load model file")
    for _ in range(1):
        model.fit(x, y, n_epoch=1, validation_set=0.1, shuffle=True,
                  show_metric=True, batch_size=64, snapshot_step=5000,
                  snapshot_epoch=False, run_id="alexnet_flowers")
        model.save(save_model_path)


def predict(network, save_file, test_data):
    model = tflearn.DNN(network)
    model.load(save_file)
    return model.predict(test_data)


if __name__ == "__main__":
    if os.path.isfile("./dataset.pkl"):
        x, y = load_data_from_pkl("./dataset.pkl")
    else:
        x, y = load_data("./train_list.txt", 17)
    net = normal_alexnet(17)
    train(net, x, y, "./pre_train_alexnet_model/alexnet_model_save.model")
