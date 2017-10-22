# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import os
caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import preprocess
import distance
import about_lmdb

# output
def output_features(source):
    samples = [os.path.join(source, s) for s in os.listdir(source)]
    all_samples = [os.path.join(source, s) for s in os.listdir(source)]

# source dataset lmdb
def predict_siamese_with_softmaxloss(source, caffemodel, deploy_file, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1"):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    dimension = 1
    with open("/Users/HZzone/caffe/data/mnist/t10k-images-idx3-ubyte", 'rb') as f:
        f.read(16) # skip the header
        raw_data = np.fromstring(f.read(10000 * 28*28), dtype=np.uint8)
    with open("/Users/HZzone/caffe/data/mnist/t10k-labels-idx1-ubyte", 'rb') as f:
        f.read(8) # skip the header
        labels = np.fromstring(f.read(10000), dtype=np.uint8)
    caffe_in = raw_data.reshape(10000, 1, 28, 28) * 0.00390625 # manually scale data instead of using `caffe.io.Transformer`
    data = np.zeros((1, dimension*2, IMAGE_SIZE, IMAGE_SIZE))
    # only for test LeNet
    data[0, :dimension, :, :] = caffe_in[0]
    data[0, dimension:, :, :] = caffe_in[1]
    net.blobs['pair_data'].data[...] = data
    output = net.forward()
    result = output[LAST_LAYER_NAME][0]
    print result
    print labels[0]
    print labels[1]


def ordinary_predict_two_sample(source1, source2, caffemodel, deploy_file, dimension=150, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1"):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    data = np.zeros((2, dimension, IMAGE_SIZE, IMAGE_SIZE))
    data[0, :, :, :] = preprocess.readManyDicom(source=source1, IMAGE_SIZE=IMAGE_SIZE, dimension=dimension)
    data[1, :, :, :] = preprocess.readManyDicom(source=source2, IMAGE_SIZE=IMAGE_SIZE, dimension=dimension)
    # only for test LeNet
    data = data * 0.00390625
    net.blobs['data'].data[...] = data
    output = net.forward()
    first_sample_feature = output[LAST_LAYER_NAME][0]
    second_sample_feature = output[LAST_LAYER_NAME][1]
    print distance.cosine_distnace(first_sample_feature, second_sample_feature)

def siamese_predict_two_sample(source, dimension=150, IMAGE_SIZE=227):
    pass

# source: Test dataset
def ordinary_predict_dataset(source, caffemodel, deploy_file, dimension=150, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1"):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    patch = about_lmdb.generate_siamese_dataset(source)
    samples = []
    for root, dirs, files in os.walk(source):
        for dicom_file in files:
            samples.append(root)
            break
    data = np.zeros((len(samples), dimension, IMAGE_SIZE, IMAGE_SIZE))
    for index, sample in enumerate(samples):
        path = os.path.join(source, sample)
        data[index, :, :, :] = preprocess.readManyDicom(path, IMAGE_SIZE, dimension) * 0.00390625
    print data.shape
    # only for test LeNet
    # data = data * 0.00390625
    net.blobs['data'].data[...] = data
    output = net.forward()
    features = output[LAST_LAYER_NAME]
    return features, samples, patch

if __name__ == "__main__":
    # # same
    # ordinary_predict_two_sample("/Users/HZzone/Desktop/temp/0007390273/29150000",
    #                             "/Users/HZzone/Desktop/temp/0007390273/33108983",
    #                             "/Users/HZzone/Downloads/lenet_iter_10000.caffemodel", "../ct-test/lenet.prototxt", gpu_mode=False)
    #
    # # diff
    # ordinary_predict_two_sample("/Users/HZzone/Desktop/temp/0008064128/10480000",
    #                             "/Users/HZzone/Desktop/temp/0007390273/33108983",
    #                             "/Users/HZzone/Downloads/lenet_iter_10000.caffemodel", "../ct-test/lenet.prototxt", gpu_mode=False)
    predict_siamese_with_softmaxloss("../mnist_siamese_with_softmax_loss/mnist_siamese_test_leveldb", "../mnist_siamese_with_softmax_loss/mnist_siamese_train_softmaxloss_iter_10000.caffemodel",
                                     "../mnist_siamese_with_softmax_loss/mnist_siamese_two_branch.prototxt", gpu_mode=False, IMAGE_SIZE=28, LAST_LAYER_NAME="classifier")
