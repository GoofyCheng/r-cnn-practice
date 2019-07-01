import cv2
import selectivesearch
import os
import pickle as pkl
import numpy as np


def intersection_area(a_xmin, a_xmax, a_ymin, a_ymax, b_xmin, b_xmax, b_ymin, b_ymax):
    lx = abs((a_xmin + a_xmax) / 2 - (b_xmin + b_xmax) / 2)
    ly = abs((a_ymin + a_ymax) / 2 - (b_ymin + b_ymax) / 2)
    a_x = abs(a_xmin - a_xmax)
    a_y = abs(a_ymin - a_ymax)
    b_x = abs(b_xmin - b_xmax)
    b_y = abs(b_ymin - b_ymax)
    if lx >= (a_x + b_x) / 2 or ly >= (a_y + b_y) / 2:
        return 0
    x = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
    y = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
    area = x * y
    return area


def iou(vec, vec1):
    vertice1 = [vec[0], vec[1], vec[0] + vec[2], vec[1] + vec[3]]
    vertice2 = [vec1[0], vec1[1], vec1[0] + vec1[2], vec1[1] + vec1[3]]
    area = intersection_area(vertice1[0], vertice1[2], vertice1[1], vertice1[3],
                             vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    a1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
    a2 = (vertice2[2] - vertice1[0]) * (vertice2[3] - vertice2[1])
    return float(area) / (a1 + a2 - area)


def img_cutting(img, vec):
    x_min = vec[0]
    x_max = vec[0] + vec[2]
    y_min = vec[1]
    y_max = vec[1] + vec[3]
    return img[x_min:x_max, y_min:y_max, :]


def preprocess_data(file_path, num_classes, save_path="./select_img_data",
                    svm=False, save=False, img_size=224, threshold=0.5):
    with open(file_path, "r") as f:
        lines = f.readlines()
    i = 0
    for line in lines:
        labels = []
        img_datas = []
        temp = line.strip().split(" ")
        img_path = temp[0]
        img_label = int(temp[1])
        img_rect = [int(i) for i in temp[2].split(",")]
        img_data = cv2.imread(img_path)
        # img_data_np = np.array(img_data, dtype="float32")
        select_img_datas, select_img_rects = selectivesearch.selective_search(
            img_data, scale=500, sigma=0.9, min_size=10)
        rects = set()
        for r in select_img_rects:
            if r["rect"] in rects:
                continue
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue
            x, y, w, h = r["rect"]
            if w == 0 or h == 0:
                continue
            select_img_data = img_cutting(img_data, r["rect"])
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
            label = np.zeros(num_classes)
            iou_val = iou(r["rect"], img_rect)
            if svm:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(img_label)
            else:
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[img_label] = 1
                labels.append(label)
        print("\rreading file: %d" % i, end='', flush=True)
        i += 1
        if save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open((os.path.join(save_path, img_path.split('/')[-1].split('.')[0].strip()) + '.pkl'), "wb") as f:
                pkl.dump((img_datas, labels), f)


def load_from_rcnn_datasetpkl(save_path):
    data_list = os.listdir(save_path)
    img_datas = []
    labels = []
    # print(data_list)
    for file in data_list:
        with open(save_path + "/" + file, "rb") as f:
            img_data, label = pkl.load(f)
            img_datas.extend(img_data)
            labels.extend(label)
    return img_datas, labels
