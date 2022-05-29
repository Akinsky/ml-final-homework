import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np


class MusicData:
    def __init__(self):
        self.df = self.load_dataset()
        text_columns = ['instance_id', 'track_name', 'obtained_date']
        self.df.drop(text_columns, axis=1, inplace=True)

        categorical_columns = ['key', 'mode', 'music_genre', 'artist_name']
        self.df[categorical_columns] = self.df[categorical_columns].astype('category')
        self.df = pd.get_dummies(self.df, columns=categorical_columns)  # OHE

        # old_df = self.df
        # self.df['artist_name'] = self.df['artist_name'].astype('category')
        # self.df['artist_name'] = self.df['artist_name'].cat.codes

        self.df = self.normalize_features(self.df)

        # self.df['artist_name'] = old_df['artist_name'].astype('category')
        # self.df['artist_name'] = self.df['artist_name'].fillna(0)

        prediction_feature = 'music_genre'
        filter_col = [col for col in self.df if col.startswith(prediction_feature)]
        self.X = self.df.drop(filter_col, axis=1)
        self.y = self.df[filter_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20,
                                                                                random_state=1)

    def load_dataset(self):
        self.df = pd.read_csv('dataset/music_genre.csv', na_values="?")  # Load the dataset and mark ?'s as null.
        self.df.dropna(inplace=True)  # Drop all null rows.
        self.df.drop_duplicates(inplace=True)  # Drop duplicate rows.
        self.df = shuffle(self.df, random_state=1)  # Shuffle the dataset.
        return self.df

    def get_x_y(self):
        return self.X, self.y

    def get_dataframe(self):
        return self.df

    def get_x_y_training(self):
        return self.X_train, self.y_train

    def get_x_y_test(self):
        return self.X_test, self.y_test

    def get_x_y_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def print_data(self):
        print(tabulate(self.df.sample(10, random_state=1), tablefmt='psql'))
        return None

    def get_y_as_single_column(self):
        self.df = self.load_dataset()
        prediction_feature = 'music_genre'
        self.df[prediction_feature] = self.df[prediction_feature].astype('category')
        self.df[prediction_feature] = self.df[prediction_feature].cat.codes
        filter_col = [col for col in self.df if col.startswith(prediction_feature)]
        X = self.df.drop(filter_col, axis=1)
        y = self.df[filter_col]

        # scaler = MinMaxScaler()
        # scaler.fit(y)
        # scaled = scaler.transform(y)
        # y = pd.DataFrame(scaled, columns=y.columns)
        _, _, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        return y_train, y_test

    def get_y_categorical(self):
        self.df = self.load_dataset()
        prediction_feature = 'music_genre'
        self.df[prediction_feature] = self.df[prediction_feature].astype('category')
        filter_col = [col for col in self.df if col.startswith(prediction_feature)]
        X = self.df.drop(filter_col, axis=1)
        y = self.df[filter_col]

        _, _, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        return y_train, y_test

    def perform_ohe(self):
        pass

    def normalize_features(self, df):
        scaler = MinMaxScaler()
        scaler.fit(df)
        scaled = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)
        return scaled_df


data = MusicData()
X_train, _, _,_ = data.get_x_y_split()
print("Current columns:",len(X_train.columns))