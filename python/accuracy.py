# import predict
import distance
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random
import threading
import datetime
import time
import array
import sample
import accuracy

'''
this file should be used in python3,
which has used cython to accelerate the speed of calcuate accuray of different threshold.
'''


totals = 100000
dic = {}

class MyThread(threading.Thread):

    def __init__(self, threshold, _same_distance, _diff_distance):
        threading.Thread.__init__(self)
        self.threshold = threshold
        self._same_distance = _same_distance
        self._diff_distance = _diff_distance

    def run(self):
        begin = datetime.datetime.now()
        acc = sample.clip(array.array('d', self._same_distance), self.threshold, array.array('d', self._diff_distance))
        accuracy.dic[self.threshold] = acc
        end = datetime.datetime.now()
        print(end - begin, acc)




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
        features = [line.strip("\n").split(" ") for line in f.readlines()]
    '''
    Temporary generate sequence
    '''
    ###################
    _same = {}
    _diff = {}
    _same_distance = []
    _diff_distance = []
    for i in range(int(totals/2)):
        while True:
            x1 = random.randint(0, len(features)-1)
            x2 = random.randint(0, len(features)-1)
            if not (x1, x2) in _same and features[x1][0] == features[x2][0]:
                _same[(x1, x2)] = ''
                break
        while True:
            x1 = random.randint(0, len(features)-1)
            x2 = random.randint(0, len(features)-1)
            if not (x1, x2) in _diff and features[x1][0] != features[x2][0]:
                _diff[(x1, x2)] = ''
                break
        print(i)
    ###################
    #### get the distances
    for x in _same.keys():
        s1, s2 = x
        d = distance.cosine_distnace(np.array(list(map(float, features[s1][2:]))),
                                     np.array(list(map(float, features[s2][2:]))))
        _same_distance.append(d)
        print(x)
    for x in _diff.keys():
        s1, s2 = x
        d = distance.cosine_distnace(np.array(list(map(float, features[s1][2:]))),
                                     np.array(list(map(float, features[s2][2:]))))
        print(x)
        _diff_distance.append(d)
    print("get the distances complete!")
    ####################
    x_values = pylab.arange(-1.0, 1.01, 0.01)
    y_values = []
    threads = []
    for threshold in x_values:
        threads.append(MyThread(threshold, _same_distance, _diff_distance))
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
    plot_accuracy("features.txt", "")

