# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import os
caffe_root = '/home/hzzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import preprocess
from itertools import combinations
import random
import distance
import random

def generate_ordinary_lmdb(source, target, IMAGE_SIZE=227):
    env = lmdb.Environment(target, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for label, person in enumerate(os.listdir(source)):
            person_dir = os.path.join(source, person)
            one_person_samples = os.listdir(person_dir)
            for im_files in one_person_samples:
                s = os.path.join(person_dir, im_files)
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 3
                datum.height = IMAGE_SIZE
                datum.width = IMAGE_SIZE
                sample = preprocess.process(s, IMAGE_SIZE)
                datum.data = sample.tobytes()
                datum.label = label
                str_id = "%s" % s
                print str_id
                txn.put(str_id, datum.SerializeToString())
            print "--------"

# source: the dataset folder
def generate_siamese_lmdb(source, target, IMAGE_SIZE=227):
    env = lmdb.Environment(target, map_size=int(1e12), writemap=True)
    dataset = generate_siamese_dataset(source, totals=250000)
    _same = dataset[0]
    _diff = dataset[1]
    random.shuffle(_same)
    random.shuffle(_diff)
    _same = [x.extend(1) for x in _same]
    _diff = [x.extend(0) for x in _diff]
    samples = []
    samples.extend(_same)
    samples.extend(_diff)
    random.shuffle(samples)
    random.shuffle(samples)
    with env.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        dimension = 3
        datum.channels = dimension
        datum.height = IMAGE_SIZE
        datum.width = IMAGE_SIZE
        sample = np.zeros((2*dimension, IMAGE_SIZE, IMAGE_SIZE))
        index = 0
        for one_sample in samples:
            label = samples[-1]
            sample[:dimension, :, :] = preprocess.process(one_sample[0], IMAGE_SIZE)
            sample[dimension:, :, :] = preprocess.process(one_sample[1], IMAGE_SIZE)
            datum.data = sample.tobytes()
            datum.label = label
            str_id = "%8d" % index
            txn.put(str_id, datum.SerializeToString())
            index = index + 1
            print index, one_sample

# generate dataset path
# combines the samples, return _same and _diff
def generate_siamese_dataset(source, totals=500000):
    all_samples = []
    _same = []
    _diff = []
    for label, person in enumerate(os.listdir(source)):
        person_dir = os.path.join(source, person)
        one_person_samples = os.listdir(person_dir)
        for dicom_files in one_person_samples:
            sample = os.path.join(person_dir, dicom_files)
            all_samples.append(sample)
    print "hello jelly"
    # sample_combinations = list(combinations(all_samples, 2))
    # sample_combinations = list(combinations(range(len(all_samples)), 2))
    # print "hello jelly"
    # for one_comination in sample_combinations:
    #     if os.path.dirname(all_samples[one_comination[0]]) == os.path.dirname(all_samples[one_comination[1]]):
    #         _same.append(one_comination)
    #     else:
    #         _diff.append(one_comination)
    # print "hello jelly"
    # random.shuffle(_diff)
    # random.shuffle(_same)
    # print len(_diff)
    # # return _same[:totals/2], _diff[:totals/2]
    # temp1 = [(all_samples[x], all_samples[y]) for x, y in _same[:totals/2]]
    # temp2 = [(all_samples[x], all_samples[y]) for x, y in _diff[:totals/2]]
    for i in range(totals/2):
        while True:
            x1 = random.randint(0, len(all_samples)-1)
            x2 = random.randint(0, len(all_samples)-1)
            if (x1, x2) not in _same and os.path.dirname(all_samples[x1]) == os.path.dirname(all_samples[x2]):
                _same.append((x1, x2))
                break
        while True:
            x1 = random.randint(0, len(all_samples)-1)
            x2 = random.randint(0, len(all_samples)-1)
            if (x1, x2) not in _diff and os.path.dirname(all_samples[x1]) != os.path.dirname(all_samples[x2]):
                _diff.append((x1, x2))
                break
        print i
    temp1 = [(all_samples[x], all_samples[y]) for x, y in _same]
    temp2 = [(all_samples[x], all_samples[y]) for x, y in _diff]
    return temp1, temp2

if __name__ == "__main__":
    # generate_ordinary_lmdb("../CASIA-WebFace", "/home/hzzone/1tb/data/train_lmdb", IMAGE_SIZE=227)
    # generate_ordinary_lmdb("../lfw", "/home/hzzone/1tb/data/test_lmdb", IMAGE_SIZE=227)
    # generate_siamese_lmdb("../CASIA-WebFace", "/home/hzzone/1tb/data/siamese_train_227_lmdb", IMAGE_SIZE=227)
    generate_siamese_lmdb("../CASIA-WebFace", "/home/hzzone/1tb/data/siamese_train_lmdb", IMAGE_SIZE=227)
    # generate_siamese_lmdb("../lfw", "/home/hzzone/1tb/data/test_lmdb", IMAGE_SIZE=227)
