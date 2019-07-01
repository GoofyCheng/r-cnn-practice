import selectivesearch
import os
import cv2
import joblib
import tflearn
import skimage
from sklearn import svm
import fine_tune_preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cnn_model import *


def prepare_img(img_path, img_size=224):
    img_data = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
                       img_data, scale=500, sigma=0.9, min_size=10)
    rects = set()
    img_datas = []
    vertices = []
    for r in regions:
        if r["rect"] in rects:
            continue
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        x, y, w, h = r["rect"]
        if w == 0 or h == 0:
            continue
        select_img_data = prep.img_cutting(img_data, r["rect"])
        if len(select_img_data) == 0:
            continue
        [a, b, c] = np.shape(select_img_data)
        if a == 0 or b == 0 or c == 0:
            continue
        rects.add(r["rect"])
        try:
            select_img_data_resize = cv2.resize(select_img_data, (img_size, img_size), cv2.INTER_CUBIC)
        except:
            print(select_img_data.shape)
        select_img_data_resize_np = np.array(select_img_data_resize, dtype="float32")
        img_datas.append(select_img_data_resize_np)
        vertices.append(r['rect'])
    return img_datas, vertices


# Load training images
def generate_single_svm_train(train_file):
    print(train_file)
    save_path = train_file.rsplit('.', 1)[0].strip()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path)
    if len(os.listdir(save_path)) == 0:
        print("reading %s's svm dataset" % train_file.split('\\')[-1])
        prep.preprocess_data(train_file, 2, save_path, threshold=0.3, svm=True, save=True)
        print("")
    print("restoring svm dataset")
    images, labels = prep.load_from_rcnn_datasetpkl(save_path)
    return images, labels


# Construct cascade svms
def train_svms(train_file_folder, model):
    files = os.listdir(train_file_folder)
    svms = []
    for train_file in files:
        if train_file.split('.')[-1] == 'txt':
            x, y = generate_single_svm_train(os.path.join(train_file_folder, train_file))
            train_features = []
            for ind, i in enumerate(x):
                # extract features
                feats = model.predict([i])
                train_features.append(feats[0])
            print("feature dimension")
            print(np.shape(train_features))
            # SVM training
            clf = svm.LinearSVC(max_iter=8000)
            print("fit svm")
            clf.fit(train_features, y)
            svms.append(clf)
            joblib.dump(clf, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm.pkl'))
    return svms


def py_cpu_nms(dets, thresh=0.3):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def show_rect(img_path, verts):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in verts:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    train_file_folder = "./svm_train"
    img_path = './17flowers/jpg/7/image_0591.jpg'
    # img_path = './17flowers/jpg/16/image_1339.jpg'
    fine_tune_model_path = "./fine_tune_model/fine_tune_model_save.model"
    imgs, verts = prepare_img(img_path)
    show_rect(img_path, verts)

    net = modify_alexnet()
    model = tflearn.DNN(net)
    model.load(fine_tune_model_path)
    svms = []
    for file in os.listdir(train_file_folder):
        if file.split('_')[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(train_file_folder, file)))
    if len(svms) == 0:
        svms = train_svms(train_file_folder, model)
    print("Done fitting svms")
    features = model.predict(imgs)
    print("predict image:")
    print(np.shape(features))
    results = []
    results_label = []
    count = 0
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            pred_prob = svm.decision_function([f.tolist()])
            # not background
            if pred[0] != 0:
                # print(pred_prob[0])
                rect = list(verts[count])
                # print(verts[count], r_l)
                rect.append(pred_prob[0])
                # print(rect)
                results.append(rect)
                results_label.append(pred[0])
        count += 1
    # print(results)
    results_np = np.array(results, dtype="float32")
    # print(results_np)
    # print(results_np.shape)
    keep_results_index = py_cpu_nms(results_np)
    keep_results = results_np[keep_results_index].tolist()
    print(keep_results)
    results_rects = [i[:4] for i in results]
    keep_rects = [i[:4] for i in keep_results]
    print("result:")
    print(results_rects)
    print("keep_rects:")
    print(keep_rects)
    print("result label:")
    print(results_label)
    show_rect(img_path, results_rects)
    show_rect(img_path, keep_rects)
    show_rect(img_path, keep_rects[:1])










