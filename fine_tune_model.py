import tflearn
import os
from cnn_model import *
from fine_tune_preprocessing import *


def fine_tune_alexnet(network, x, y, normal_model_path, fine_tune_model_path):
    model = tflearn.DNN(network, checkpoint_path="fine_tune_alexnet",
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="fine_tune_output")
    if os.path.isfile(fine_tune_model_path + ".index"):
        model.load(fine_tune_model_path)
        print("load fine_tune_model")
    elif os.path.isfile(normal_model_path + ".index"):
        model.load(normal_model_path)
        print("load normal_model")
    else:
        raise ValueError("can not find model params")
    for _ in range(1):
        model.fit(x, y, n_epoch=1, validation_set=0.1, show_metric=True, shuffle=True,
                  batch_size=64, snapshot_step=5000, snapshot_epoch=False,
                  run_id="fine_tune_alextnet")
        model.save(fine_tune_model_path)


if __name__ == "__main__":
    if os.path.exists("./select_img_data"):
        x, y = load_from_rcnn_datasetpkl("select_img_data")
    else:
        preprocess_data("fine_tune_list.txt", 3, save=True)
        x, y = load_from_rcnn_datasetpkl("select_img_data")
    net = normal_alexnet(3, restore=False)
    model = fine_tune_alexnet(net, x, y, "./pre_train_alexnet_model/alexnet_model_save.model",
                              "./fine_tune_model/fine_tune_model_save.model")
