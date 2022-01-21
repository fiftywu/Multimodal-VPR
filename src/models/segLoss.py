import numpy as np


def fast_hist(a, b, n):
    # a is groundtruch map
    # b is pred segmap
    # n is num of classes
    k = (a >= 0) & (a < n)
    return np.bincount(n*a[k].astype(int)+b[k],
                       minlength=n**2).reshape(n, n)

class SegMetrics():
    def __init__(self, gtmap, pred_map, nclasses):
        self.gt = gtmap
        self.pred = pred_map
        self.nums = nclasses
        self.hist = fast_hist(self.gt.flatten(),
                              self.pred.flatten(),
                              self.nums)

    def over_acc(self):
        return np.diag(self.hist).sum() / self.hist.sum()
        # one value

    def perclass_acc(self):
        return np.diag(self.hist) / self.hist.sum(1)
        # one array

    def perclass_iu(self):
        return np.diag(self.hist) / (self.hist.sum(1) +
                                     self.hist.sum(0) -
                                     np.diag(self.hist))
        # one array

    def fw_perclass_iu(self):
        iu = self.perclass_iu()
        freq = self.hist.sum(1) / self.hist.sum()
        return freq * iu

    def mean_metric(self):
        return [self.over_acc(),
                np.nanmean(self.perclass_acc()),
                np.nanmean(self.perclass_iu()),
                np.nansum(self.fw_perclass_iu())]

    def perclass_metric(self):
        return [[self.over_acc() for _ in range(len(self.perclass_acc()))],
                self.perclass_acc(),
                self.perclass_iu(),
                self.fw_perclass_iu()]

if __name__ == '__main__':
    label = np.array([[0,1], [0,0]])
    output = np.array([[1,1], [1,0]])
    nclasses = 2

    label = np.array([[1,2], [1,1]])
    output = np.array([[2,2], [2,1]])
    nclasses = 3
    Metrics = SegMetrics(label, output, nclasses)
    print(Metrics.over_acc())
    print(Metrics.perclass_acc())
    print(Metrics.perclass_iu())
    print(Metrics.fw_perclass_iu())
    print(Metrics.mean_metric())