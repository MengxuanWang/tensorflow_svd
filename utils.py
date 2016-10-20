import random
from itertools import islice
import numpy as np

# Generate a batch iterator for a datasets.
def batch_iter(data, batch_size, num_epochs):
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffle_data = data[shuffle_indices]
        for batch_num in xrange(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1) * batch_size, data_size)
            yield shuffle_data[start_index: end_index]

def dot_product(vec1, vec2):
    assert len(vec1) == len(vec2)
    return sum(vec1[i]*vec2[i] for i in range(len(vec1)))

def avg_rating(ratings):
    return np.mean(ratings)

def load_ratings(trainfile, nsample=0):
    trains = []
    with open(trainfile) as f:
        for line in islice(f, 1, None):
            u, v, r, t = [int(item) for item in line.strip().split(',')]
            trains.append([u, v, r, t])

    trains = np.array(trains)
    if nsample:
        print "number of train data before sample: ", len(trains)
        indices = np.random.choice(np.arange(len(trains)), nsample, replace=False)
        trainrats = np.array(trains[indices])
        print "number of train data after sample: ", len(trainrats)
        trains = trainrats

    nuser = np.max(trains[:, 0]) + 1
    nitem = np.max(trains[:, 1]) + 1
    mu = avg_rating(trains[:, 2])
    return [trains, nuser, nitem, mu]
