import sample
import numpy as np
_same_distance = np.array([4, 1, 1, 5, 3], dtype=float)
_diff_distance = np.array([1, 1, 1, 6, 3], dtype=float)
# should output 0.6
print(sample.clip(_same_distance, 2.0, _diff_distance))
print("test clip")

_same_distance = np.array([4, 1, 1, 5, 3], dtype=float)
_diff_distance = np.array([1, 1, 1, 6, 3], dtype=float)
# should output 0.6
print(sample.clip(_same_distance, 2.0, _diff_distance))
