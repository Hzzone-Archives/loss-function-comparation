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

# source: Test dataset, generate dataset's features(person name, file name, features)
def output_features_of_dataset(source, caffemodel, deploy_file, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1", batch_size=240, save_file_path="./features.txt"):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    samples = []
    for dir_name in os.listdir(source):
        one_person_dir = os.path.join(source, dir_name)
        for file_name in os.listdir(source):
            one_person_pic_path = os.path.join(one_person_dir, file_name)
            samples.append((dir_name, file_name, one_person_pic_path))
    data = np.zeros((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE))
    with open(save_file_path, "w") as f:
        for index, sample in enumerate(samples):
            t = index % batch_size
            data[t, :, :, :] = preprocess.process(sample[2], IMAGE_SIZE)
            if t == 0:
                net.blobs['data'].data[...] = data
                output = net.forward()
                features = output[LAST_LAYER_NAME]
                lines = ["%s %s %s\n" % (s[0][0], s[0][1], " ".join(s[1])) for s in zip(samples[index-50:index], features)]
                f.writelines(lines)

if __name__ == "__main__":
    pass
