import math

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.avg = 0.0
        self.m2 = 0.0  # The "M2" term in Welford's algorithm

    def update(self, val, n=1):
        self.count += n

        # Update mean and M2 term for Welford's algorithm
        delta = val - self.avg
        self.avg += delta / self.count
        delta2 = val - self.avg
        self.m2 += delta * delta2

    def average(self):
        return self.avg if self.count > 0 else 0.0

    def variance(self):
        return self.m2 / self.count if self.count > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())
