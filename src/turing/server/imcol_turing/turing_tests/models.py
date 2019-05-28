from django.db import models


class TuringTest(models.Model):
    image = models.IntegerField()
    set = models.IntegerField()
    is_true = models.BooleanField()
    is_correct = models.BooleanField()
    time = models.DurationField()
    ip_address = models.CharField(max_length=128)
