import time
class AverageMeter_base(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.time_now = self.timer()
        self.time_pre = self.time_now
        self.time_all = self.time_now - self.time_pre

        self.loss = 0
        self.count = 0
        self.avg_loss = 0

    def timer(self):
        return time.time()

    def update(self, loss, count):
        self.time_now = self.timer()
        self.time_all += (self.time_now - self.time_pre)
        self.time_pre = self.time_now

        self.count += count
        self.loss += loss * count
        self.avg_loss = self.loss/self.count