from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def predict_popularity(acousticness, danceability, energy, instrumentalness, valence, speechiness, loudness, liveness):
    if valence >0.494 and acousticness<0.000411 and danceability>0.126 and energy>0.675 and instrumentalness<0.884 and liveness>0.0727 and speechiness<0.0445 and loudness>-8.547:
        return True
    else:
        return False