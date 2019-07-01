from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# 完整alexnet网络,增加最后一层参数是否读取保存文件的参数restore
def normal_alexnet(num_classes, img_size=224, channel=3, restore=True):
    network = input_data(shape=[None, img_size, img_size, channel])
    network = conv_2d(network, 96, 11, strides=4, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation="relu")
    network = conv_2d(network, 384, 3, activation="relu")
    network = conv_2d(network, 256, 3, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation="tanh")
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation="tanh")
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, restore=restore, activation="softmax")
    network = regression(network, optimizer="momentum",
                         loss="categorical_crossentropy", learning_rate=1e-3)
    return network


# 去除最后一层softmax全连接层的alexnet
def modify_alexnet(img_size=224, channel=3):
    network = input_data(shape=[None, img_size, img_size, channel])
    network = conv_2d(network, 96, 11, strides=4, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation="relu")
    network = conv_2d(network, 384, 3, activation="relu")
    network = conv_2d(network, 256, 3, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation="tanh")
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation="tanh")
    network = regression(network, optimizer="momentum",
                         loss="categorical_crossentropy", learning_rate=1e-3)
    return network


