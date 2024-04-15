from django.shortcuts import render, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from django.http import HttpResponse
from io import BytesIO
from sklearn.tree import DecisionTreeClassifier
import base64
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from django.http import JsonResponse
import joblib
from .models import Song
from .inference_engine import predict_popularity

# Specify the path to your CSV file

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if (username == 'MusicWorld'):
            if(password == 'music123'):
                return redirect('dashboard')
        else:
            return redirect('login')
    return render(request, 'MusicWorld/login.html')


def dashboard(request):
    return render(request, 'MusicWorld/dashboard.html')
def first(request):
    return render(request, 'MusicWorld/first.html')
def contactus(request):
    return render(request, 'MusicWorld/contactus.html')
def c(request):
    return render(request, 'MusicWorld/c.html')
def cm(request):
    return render(request, 'MusicWorld/cm.html')
def pl(request):
    return render(request, 'MusicWorld/pl.html')
def fs(request):
    return render(request, 'MusicWorld/fs.html')
def aboutus(request):
    return render(request, 'musicworld/aboutus.html')
def vr(request):
    return render(request, 'MusicWorld/vr.html')
def dt_selection(request):
    df = pd.read_csv(r'MusicWorld\static\dataset\SpotifySongPolularityAPIExtracttt.csv', encoding='ISO-8859-1')
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    df['artist_name'] = label_encoder.fit_transform(df['artist_name'])
    df['track_name'] = label_encoder.fit_transform(df['track_name'])
    df['track_id'] = label_encoder.fit_transform(df['track_id'])
    all_features = ['acousticness', 'tempo', 'energy', 'instrumentalness', 'valence', 'speechiness', 'duration_ms', 'loudness', 'liveness', 'danceability', 'artist_name']
    target = ['popularity']
    X = df[all_features]
    y = df[target]
    threshold = 1 
    y_class = (y > threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, y_train)
    feature_importances = tree.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(all_features, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Decision Tree Feature Importance')
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()
    image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    context = {'image_base64': image_base64}
    return render(request, 'musicworld/dt_selection.html', context)

def knn(request):
    df = pd.read_csv(r'MusicWorld\static\dataset\SpotifySongPolularityAPIExtracttt.csv', encoding='ISO-8859-1')
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    df['artist_name'] = label_encoder.fit_transform(df['artist_name'])
    df['track_name'] = label_encoder.fit_transform(df['track_name'])
    df['track_id'] = label_encoder.fit_transform(df['track_id'])
    all_features = ['acousticness', 'tempo', 'energy', 'instrumentalness', 'valence', 'speechiness', 'duration_ms', 'loudness', 'liveness', 'danceability', 'artist_name']
    target = ['popularity']
    X = df[all_features]
    y = df[target]
    threshold = 1  # Set your threshold for popularity
    y_class = (y > threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)
    tree = DecisionTreeRegressor(random_state=42)
    feature_selector = SelectFromModel(tree)
    feature_selector.fit(X_train, y_train)
    selected_features = X_train.columns[feature_selector.get_support()]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    k = 3  # You can choose any suitable value for k
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_selected, y_train)
    y_pred = knn_classifier.predict(X_test_selected)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy_percentage = accuracy_score(y_test, y_pred) * 100
    image_stream = BytesIO()
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Popular', 'Popular'], yticklabels=['Not Popular', 'Popular'])
    plt.title("Confusion Matrix (Testing Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(image_stream, format='png')
    plt.close()
    context = {'image_base64': base64.b64encode(image_stream.getvalue()).decode('utf-8'),
               'accuracy_percentage': f'{accuracy_percentage:.2f}%'}
    return render(request, 'musicworld/knn.html', context)

def dt_class(request):
    df = pd.read_csv(r'MusicWorld\static\dataset\SpotifySongPolularityAPIExtracttt.csv', encoding='ISO-8859-1')
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    df['artist_name'] = label_encoder.fit_transform(df['artist_name'])
    df['track_name'] = label_encoder.fit_transform(df['track_name'])
    df['track_id'] = label_encoder.fit_transform(df['track_id'])
    all_features = ['acousticness', 'tempo', 'energy', 'instrumentalness', 'valence', 'speechiness', 'duration_ms', 'loudness', 'liveness', 'danceability', 'artist_name']
    target = ['popularity']
    X = df[all_features]
    y = df[target]
    threshold = 1  # Set your threshold for popularity
    y_class = (y > threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)
    tree_classifier = DecisionTreeClassifier(random_state=42)
    tree_classifier.fit(X_train, y_train)
    y_pred = tree_classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy_percentage = accuracy_score(y_test, y_pred) * 100
    image_stream = BytesIO()
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Popular', 'Popular'], yticklabels=['Not Popular', 'Popular'])
    plt.title("Confusion Matrix (Testing Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(image_stream, format='png')
    plt.close()
    context = {'image_base64': base64.b64encode(image_stream.getvalue()).decode('utf-8'),
               'accuracy_percentage': f'{accuracy_percentage:.2f}%'}
    return render(request, 'musicworld/dt_class.html', context)


def nb(request):
    df = pd.read_csv(r'MusicWorld\static\dataset\SpotifySongPolularityAPIExtracttt.csv', encoding='ISO-8859-1')
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    df['artist_name'] = label_encoder.fit_transform(df['artist_name'])
    df['track_name'] = label_encoder.fit_transform(df['track_name'])
    df['track_id'] = label_encoder.fit_transform(df['track_id'])
    all_features = ['acousticness', 'tempo', 'energy', 'instrumentalness', 'valence', 'speechiness', 'duration_ms', 'loudness', 'liveness', 'danceability', 'artist_name']
    target = ['popular']
    X = df[all_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy_percentage = accuracy_score(y_test, y_pred) * 100
    image_stream = BytesIO()
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Popular', 'Popular'], yticklabels=['Not Popular', 'Popular'])
    plt.title("Confusion Matrix (Testing Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(image_stream, format='png')
    plt.close()
    context = {'image_base64': base64.b64encode(image_stream.getvalue()).decode('utf-8'),
               'accuracy_percentage': f'{accuracy_percentage:.2f}%'}
    return render(request, 'musicworld/nb.html', context)

def display_records(request):
    csv_path = 'MusicWorld/static/dataset/SpotifySongPolularityAPIExtracttt.csv'
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    subset_df = df.head(1000)
    records = subset_df.to_dict(orient='records')
    context = {'records': records}
    return render(request, 'musicworld/display_records.html', context)


def popular(request):
    csv_path = 'MusicWorld/static/dataset/SpotifySongPolularityAPIExtracttt.csv'
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    subset_df = df.head(10000)
    records = subset_df.to_dict(orient='records')
    context = {'records': records}
    return render(request, 'musicworld/popular.html', context)

def not_popular(request):
    csv_path = 'MusicWorld/static/dataset/SpotifySongPolularityAPIExtracttt.csv'
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    subset_df = df.head(5000)
    records = subset_df.to_dict(orient='records')
    context = {'records': records}
    return render(request, 'musicworld/not_popular.html', context)

def display_selected(request):
    csv_path = 'MusicWorld/static/dataset/SpotifySongPolularityAPIExtracttt.csv'
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    subset_df = df.head(1000)
    records = subset_df.to_dict(orient='records')
    context = {'records': records}
    return render(request, 'musicworld/display_selected.html', context)

def predict_song(request):
    if request.method == 'POST':
        acousticness = float(request.POST.get('acousticness'))
        danceability = float(request.POST.get('danceability'))
        energy = float(request.POST.get('energy'))
        instrumentalness = float(request.POST.get('instrumentalness'))
        valence = float(request.POST.get('valence'))
        speechiness = float(request.POST.get('speechiness'))
        loudness = float(request.POST.get('loudness'))
        liveness = float(request.POST.get('liveness'))
        is_popular = predict_popularity(acousticness, danceability, energy, instrumentalness, valence, speechiness, loudness, liveness)
        return render(request, 'MusicWorld/result.html', {'is_popular': is_popular})
    return render(request, 'MusicWorld/predict_song.html')

def result(request):
        return render(request, 'MusicWorld/result.html')