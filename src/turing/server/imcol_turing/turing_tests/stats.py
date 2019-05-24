import numpy as np

from .models import TuringTest


class SetStatistics:
    def __init__(self, set_id=-1):
        if set_id < 0:
            data = TuringTest.objects
        else:
            data = TuringTest.objects.filter(set=set_id)

        self.tp = data.filter(is_correct=True, is_true=True).count()
        self.tn = data.filter(is_correct=True, is_true=False).count()
        self.fp = data.filter(is_correct=False, is_true=False).count()
        self.fn = data.filter(is_correct=False, is_true=True).count()

        self.correct_guesses = self.tp + self.tn
        self.incorrect_guesses = self.fp + self.fn
        self.total_guesses = self.correct_guesses + self.incorrect_guesses

        self.accuracy = np.round(100. * self.correct_guesses / self.total_guesses, 2)

        self.tpr = np.round(100. * self.tp / (self.tp + self.fn), 2)
        self.tnr = np.round(100. * self.tn / (self.tn + self.fp), 2)

        self.participants = data.values('ip_address').distinct().count()

        #self.time_mean = np.round(np.mean(data.values('time')), 2)
        #self.time_std = np.std(data.values('time'))
