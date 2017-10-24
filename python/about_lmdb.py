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
# from numba import jit

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
    for x in _same:
        x.append(1)
    for x in _diff:
        x.append(0)
    samples = []
    samples.extend(_same)
    samples.extend(_diff)
    # print samples
    random.shuffle(samples)
    random.shuffle(samples)
    random.shuffle(samples)
    # print len(samples)
    # print samples
    with env.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        dimension = 3
        datum.channels = dimension
        datum.height = IMAGE_SIZE
        datum.width = IMAGE_SIZE
        sample = np.zeros((2*dimension, IMAGE_SIZE, IMAGE_SIZE))
        index = 0
        for one_sample in samples:
            print index, one_sample
            label = one_sample[-1]
            sample[:dimension, :, :] = preprocess.process(one_sample[0], IMAGE_SIZE)
            sample[dimension:, :, :] = preprocess.process(one_sample[1], IMAGE_SIZE)
            datum.data = sample.tobytes()
            datum.label = label
            str_id = "%8d" % index
            txn.put(str_id, datum.SerializeToString())
            index = index + 1

# generate dataset path
# combines the samples, return _same and _diff

# @jit
def generate_siamese_dataset(source, totals=500000):
    all_samples = []
    _same = {}
    _diff = {}
    for label, person in enumerate(os.listdir(source)):
        person_dir = os.path.join(source, person)
        one_person_samples = os.listdir(person_dir)
        for dicom_files in one_person_samples:
            sample = os.path.join(person_dir, dicom_files)
            all_samples.append(sample)
    for i in range(totals/2):
        while True:
            x1 = random.randint(0, len(all_samples)-1)
            x2 = random.randint(0, len(all_samples)-1)
            if not _same.has_key((x1, x2)) and os.path.dirname(all_samples[x1]) == os.path.dirname(all_samples[x2]):
                _same[(x1, x2)] = ''
                break
        while True:
            x1 = random.randint(0, len(all_samples)-1)
            x2 = random.randint(0, len(all_samples)-1)
            if not _diff.has_key((x1, x2)) and os.path.dirname(all_samples[x1]) != os.path.dirname(all_samples[x2]):
                # _diff.append((x1, x2))
                _diff[(x1, x2)] = ''
                break
        print i
    temp1 = [list((all_samples[x], all_samples[y])) for x, y in _same.keys()]
    temp2 = [list((all_samples[x], all_samples[y])) for x, y in _diff.keys()]
    return temp1, temp2

if __name__ == "__main__":
    # generate_ordinary_lmdb("../CASIA-WebFace", "/home/hzzone/1tb/data/train_lmdb", IMAGE_SIZE=227)
    # generate_ordinary_lmdb("../lfw", "/home/hzzone/1tb/data/test_lmdb", IMAGE_SIZE=227)
    # generate_siamese_lmdb("../CASIA-WebFace", "/home/hzzone/1tb/data/siamese_train_227_lmdb", IMAGE_SIZE=227)
    generate_siamese_lmdb("../CASIA-WebFace", "/home/hzzone/1tb/data/siamese_train_lmdb", IMAGE_SIZE=227)
    # generate_siamese_lmdb("../lfw", "/home/hzzone/1tb/data/test_lmdb", IMAGE_SIZE=227)
