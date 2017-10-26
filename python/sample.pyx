# sample.pyx (Cython)

cimport cython
import distance
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double clip(double[:] _same_distance, double threshold, double[:] _diff_distance):
    '''
    claculate the accuracy by the threshold.
    '''
    correct = 0
    totals = len(_same_distance) + len(_diff_distance)
    for x in _same_distance:
        if x >= threshold:
            correct += 1
    for x in _diff_distance:
        if x < threshold:
            correct += 1
    return float(correct)/totals

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :] get_distances(double[:, :] _same, double[:, :] _diff):
    '''
    claculate the distance.
    '''
    _same_disatance = []
    _diff_disatance = []
    for x in _same:
        _same_disatance.append(distance)
    for x in _diff:
        _diff_disatance.append(distance)
    return np.array((_same_disatance, _diff_disatance))

