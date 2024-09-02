from django.db import models

class Analysis(models.Model):
    analyzer = models.CharField(max_length=50)
    text = models.TextField()
    sentiment = models.CharField(max_length=10)
    score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.analyzer} - {self.sentiment} - {self.score} - {self.created_at}"

    
