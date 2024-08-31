from django.db import models

class Analysis(models.Model):
    text = models.TextField()
    sentiment = models.CharField(max_length=10)
    score = models.FloatField()

    
