import predict
import distance
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random
import threading
import sys
import datetime
import time
import accuracy
import logging


features = None
_same = {}
_diff = {}
totals = 100000
dic = {}

class MyThread(threading.Thread):

    def __init__(self, threshold):
        threading.Thread.__init__(self)
        self.threshold = threshold

    def run(self):
        begin = datetime.datetime.now()
        correct = 0
        index = 0
        for x in accuracy._same.keys():
            s1, s2 = x
            print index
            d = distance.cosine_distnace(np.array(map(float, accuracy.features[s1][2:])), np.array(map(float, accuracy.features[s2][2:])))
            if d >= self.threshold:
                correct = correct + 1
            index += 1
        for x in accuracy._diff.keys():
            s1, s2 = x
            print ''
            d = distance.cosine_distnace(np.array(map(float, accuracy.features[s1][2:])), np.array(map(float, accuracy.features[s2][2:])))
            if d < self.threshold:
                correct = correct + 1
            index += 1
        print "------"*2
        self.result = self.threshold, float(correct)/totals
        accuracy.dic[self.threshold] = float(correct)/totals
        end = datetime.datetime.now()
        print end - begin

    def get_result(self):
        return self.result



'''
plot accuracy map
from features.txt and test sequence
'''
def plot_accuracy(features_source, sequence_source):
    '''

    :param features_source: the features txt
    :param sequence_source: the  sequence text
    :return: None
    '''
    '''
    read features from features.txt
    '''
    with open(features_source) as f:
        accuracy.features = [line.strip("\n").split(" ") for line in f.readlines()]
    '''
    Temporary generate sequence
    '''
    ###################
    for i in range(totals/2):
        while True:
            x1 = random.randint(0, len(accuracy.features)-1)
            x2 = random.randint(0, len(accuracy.features)-1)
            if not accuracy._same.has_key((x1, x2)) and accuracy.features[x1][0]==accuracy.features[x2][0]:
                accuracy._same[(x1, x2)] = ''
                break
        while True:
            x1 = random.randint(0, len(accuracy.features)-1)
            x2 = random.randint(0, len(accuracy.features)-1)
            if not accuracy._diff.has_key((x1, x2)) and accuracy.features[x1][0]!=accuracy.features[x2][0]:
                accuracy._diff[(x1, x2)] = ''
                break
        print i
    ###################
    x_values = pylab.arange(-1.0, 1.01, 0.01)
    y_values = []
    threads = []
    for threshold in x_values:
        threads.append(MyThread(threshold))
        # y_values.append(float(correct)/totals)
    for t in threads:
        t.start()
    time.sleep(500)
    # result = [t.get_result() for t in threads]
    d = accuracy.dic
    acc = sorted(d.items(), key=lambda d:d[0])
    for temp in acc:
        y_values.append(temp[1])
    max_index = np.argmax(y_values)
    plt.title("threshold-accuracy curve")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.plot(x_values, y_values)
    plt.plot(x_values[max_index], y_values[max_index], '*', color='red', label="(%s, %s)"%(x_values[max_index], y_values[max_index]))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_accuracy("/Users/HZzone/Desktop/features.txt", "")

