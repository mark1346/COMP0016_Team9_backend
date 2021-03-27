from django.db import models
from enum import Enum
import jsonfield

# Create your models here.

class WordPair(models.Model):
    pair1 = models.CharField(max_length=100)
    pair2 = models.CharField(max_length=100)

    def __str__(self):
        return "pair1: {}, pair2: {}".format(self.pair1, self.pair2)

# Word model (For bias detect)
class Word(models.Model):
    word = models.CharField(max_length=100)

    def __str__(self):
        return self.word



class UserConfig(models.Model):
    mode = models.CharField(max_length=100)
    bias = jsonfield.JSONField()

    