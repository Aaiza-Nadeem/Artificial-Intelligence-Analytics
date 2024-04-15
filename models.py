from django.db import models

class Song(models.Model):
    acousticness = models.FloatField()
    tempo = models.FloatField()
    energy = models.FloatField()
    instrumentalness = models.FloatField()
    valence = models.FloatField()
    speechiness = models.FloatField()
    duration_ms = models.FloatField()
    loudness = models.FloatField()
    liveness = models.FloatField()
    danceability = models.FloatField()
    artist_name = models.CharField(max_length=255)
    track_name = models.CharField(max_length=255)
    track_id = models.BigIntegerField()
    popular = models.CharField(max_length=255, default='0')
    popularity = models.CharField(max_length=255, default='0')
    key = models.CharField(max_length=255, default='0')
    mode = models.CharField(max_length=255, default='0')
    time_signature = models.CharField(max_length=255, default='0')
    def __str__(self):
        return self.track_name