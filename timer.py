from time import perf_counter

class Timer:
    def __init__(self):
        self.now = perf_counter()
        self.start = self.now

    def elapsed(self):
        old_now = self.now
        self.now = perf_counter()
        return self.now - old_now

    def __str__(self):
        return str(self.total())

    def total(self):
        return perf_counter() - self.start

    def reset(self):
        self.elapsed()
