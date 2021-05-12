import random
import timeit

import keras.models
import pandas as pd
import spotipy
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from spotipy.oauth2 import SpotifyClientCredentials
from numpy import argmax


def base_model():
    model = Sequential()
    model.add(Dense(8, input_dim=9, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cleanDuplicates(tracks):
    grouped = tracks.groupby(['artist_name', 'track_name'], as_index=True).size()
    grouped[grouped > 1].count()
    tracks.drop_duplicates(subset=['artist_name', 'track_name'], inplace=True)


def dropColumns(df_audio_features):
    columns_to_drop = ['analysis_url', 'track_href', 'type', 'key', 'mode', 'time_signature', 'uri']
    df_audio_features.drop(columns_to_drop, axis=1, inplace=True)

    df_audio_features.rename(columns={'id': 'track_id'}, inplace=True)

    return df_audio_features.shape


def mergeDataframes(df_tracks, df_audio_features):
    # merge both dataframes
    # the 'inner' method will make sure that we only keep track IDs present in both datasets
    df = pd.merge(df_tracks, df_audio_features, on='track_id', how='inner')

    return df


def getRandomSearch():
    # A list of all characters that can be chosen.
    characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']

    # Gets a random character from the characters string.
    randomCharacter = characters[random.randint(0, 25)]

    # Places the wildcard character at the beginning, or both beginning and end, randomly.
    switcher = random.randint(0, 1)
    if switcher == 0:
        randomSearch = randomCharacter + '%'
    else:
        randomSearch = '%' + randomCharacter + '%'

    return randomSearch


class MusicMoodClassifier:
    def __init__(self):
        self.cid = "a006ea8174bc4689b4eb39c47b5449a1"
        self.secret = "1cca3d1fff6145fdaee72ba822e8b586"
        self.client_credentials_manager = SpotifyClientCredentials(client_id=self.cid, client_secret=self.secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        self.estimator = keras.models.load_model('ml/music_model.h5')

    def getTracks(self, query, number):
        start = timeit.default_timer()

        # create empty lists where the results are going to be stored
        artist_name = []
        track_name = []
        popularity = []
        track_id = []

        for i in range(0, number, 10):
            track_results = self.sp.search(q=query, type='track', limit=10, offset=i)
            for i, t in enumerate(track_results['tracks']['items']):
                artist_name.append(t['artists'][0]['name'])
                track_name.append(t['name'])
                track_id.append(t['id'])
                popularity.append(t['popularity'])

        stop = timeit.default_timer()
        #print('Time to run this code (in seconds):', stop - start)
        df_tracks = pd.DataFrame(
            {'artist_name': artist_name, 'track_name': track_name, 'track_id': track_id, 'popularity': popularity})
        return df_tracks

    def getAudioFeatures(self, tracks):
        # again measuring the time
        start = timeit.default_timer()

        # empty list, batchsize and the counter for None results
        rows = []
        batchsize = 100
        none_counter = 0

        for i in range(0, len(tracks['track_id']), batchsize):
            batch = tracks['track_id'][i:i + batchsize]
            feature_results = self.sp.audio_features(batch)
            for i, t in enumerate(feature_results):
                if t == None:
                    none_counter = none_counter + 1
                else:
                    rows.append(t)

        #print('Number of tracks where no audio features were available:', none_counter)

        stop = timeit.default_timer()
        #print('Time to run this code (in seconds):', stop - start)
        df_audio_features = pd.DataFrame.from_dict(rows, orient='columns')
        return df_audio_features

    def getTypicalTracks(self, typicalMood):
        if typicalMood == 1:
            test = self.getTracks(getRandomSearch(), 501)
        else:
            test = self.getTracks(getRandomSearch(), 50)
        test_features = self.getAudioFeatures(test)
        dropColumns(test_features)
        df_test = mergeDataframes(test, test_features)
        test_col_features = df_test.columns[4:13]
        df_test_features = df_test[test_col_features]
        df_test_features = MinMaxScaler().fit_transform(df_test_features)
        mood_preds_test = self.estimator.predict(df_test_features)
        mood_preds = []
        for i in range(len(mood_preds_test)):
            mood_preds.append(argmax(mood_preds_test[i]))
        IDs = test['track_id']
        names = test['track_name']
        results = []
        final_results = []
        if typicalMood == 1:
            for x in range(100):
                if mood_preds[x] == typicalMood:
                    results.append([names[x], IDs[x]])
        else:
            for x in range(50):
                if mood_preds[x] == typicalMood:
                    results.append([names[x], IDs[x]])

        result = self.sp.track(results[0][1])
        ResponseResult=[result['name'],"https://open.spotify.com/track/"+result['id'], result['album']['images'][0]['url']]
        
        return ResponseResult
