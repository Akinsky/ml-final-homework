import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import shuffle


class MusicData:
    def __init__(self, dtree=False):
        self.df = self.load_dataset()

        text_columns = ['instance_id', 'track_name', 'obtained_date', 'artist_name', 'index']
        self.df.drop(text_columns, axis=1, inplace=True)

        self.numeric_features = self.df.drop(['mode', 'music_genre', 'key'], axis=1)

        # self.get_plots()  ### Commented for faster executing purposes. Uncomment to generate plots for the first time.

        categorical_columns = ['mode', 'music_genre', 'key']
        self.df[categorical_columns] = self.df[categorical_columns].astype('category')
        self.y_names = list(self.df['music_genre'].cat.categories)

        if not dtree:
            self.df = pd.get_dummies(self.df, columns=categorical_columns)  # OHE
            # self.get_heatmap("heatmap_ohe", nums=False, linewidth=0)
        else:
            # No OHE, just turn categorical values into numerical.
            self.df[categorical_columns] = self.df[categorical_columns].apply(lambda x: x.cat.codes)

        self.populate_dataset(dtree)  # Add many new columns.

        if not dtree:
            self.df = self.normalize_features(self.df)  # No need to normalize for decision tree.
        else:
            # If we didn't apply OHE, mark categorical columns as integer so the decision tree can predict.
            self.df[categorical_columns] = self.df[categorical_columns].astype('int')

        # Returns y as the 'music_genre' column and the rest as X.
        self.X, self.y = self.split_feature_as_x_y('music_genre')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.20,
                                                                                random_state=1)

    def load_dataset(self):
        self.df = pd.read_csv(f'dataset/music_genre.csv', na_values="?")  # Load the dataset and mark ?'s as null.
        self.df['duration_ms'].replace(to_replace=-1, value=np.nan, inplace=True)
        self.df.dropna(inplace=True)  # Drop all null rows.
        self.df.drop_duplicates(inplace=True)  # Drop duplicate rows.
        self.df.reset_index(inplace=True)  # Reset indices since we deleted some rows.
        self.df = shuffle(self.df, random_state=1)  # Shuffle the dataset.
        return self.df

    def populate_dataset(self, dtree=False):
        self.df['instrumentalness'].replace(0, np.nan, inplace=True)
        self.df['instrumentalness'].fillna((self.df['instrumentalness'].median()), inplace=True)

        self.df['eng_loud'] = self.df['energy'] * self.df['loudness']  # 0.8 Relation
        self.df['eng_acstc'] = self.df['energy'] * self.df['acousticness']  # -0.8 Relation
        self.df['acstc_loud'] = self.df['acousticness'] * self.df['loudness']  # -0.7 Relation
        self.df['loud_instr'] = self.df['loudness'] * self.df['instrumentalness']  # -0.5 Relation
        self.df['loud_dance'] = self.df['loudness'] * self.df['danceability']  # 0.4 Relation

        if not dtree:
            self.df['minor_valence'] = self.df['mode_Minor'] * self.df['valence']
            self.df['major_valence'] = self.df['mode_Major'] * self.df['valence']

        self.df['energy^2'] = self.df['energy'] ** 2
        self.df['popularity^2'] = self.df['popularity'] ** 2
        self.df['valence^2'] = self.df['valence'] ** 2
        self.df['sin(energy)'] = np.sin(self.df['energy'])
        self.df['duration_cbrt'] = np.cbrt(self.df['duration_ms'])
        self.df['liveness_cos'] = np.sqrt(self.df['liveness'])

    def split_feature_as_x_y(self, prediction_feature):
        prediction_feature = 'music_genre'
        filter_col = [col for col in self.df if col.startswith(prediction_feature)]
        X = self.df.drop(filter_col, axis=1)
        y = self.df[filter_col]
        return X, y

    def get_dataframe(self):
        return self.df

    def get_x_y_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_y_as_numerical(self):

        self.df = self.load_dataset()
        prediction_feature = 'music_genre'
        self.df[prediction_feature] = self.df[prediction_feature].astype('category')
        self.df[prediction_feature] = self.df[prediction_feature].cat.codes
        filter_col = [col for col in self.df if col.startswith(prediction_feature)]
        X = self.df.drop(filter_col, axis=1)
        y = self.df[filter_col]

        _, _, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        return y_train, y_test

    def get_y_categorical(self):
        """ Returns y column as in the original dataset as categories."""
        self.df = self.load_dataset()
        prediction_feature = 'music_genre'
        self.df[prediction_feature] = self.df[prediction_feature].astype('category')
        filter_col = [col for col in self.df if col.startswith(prediction_feature)]
        X = self.df.drop(filter_col, axis=1)
        y = self.df[filter_col]

        _, _, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        return y_train, y_test

    def get_y_names(self):
        return self.y_names

    def normalize_features(self, df):
        scaler = RobustScaler()
        scaler.fit(df)
        scaled = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)
        return scaled_df

    def get_heatmap(self, filename, nums=True, linewidth=0, figsize=(16, 12)):
        """ Returns a heatmap of the dataset. """
        matplotlib.rc_file_defaults()
        plt.figure(figsize=(16, 12))  # Increase figure size to fit the heatmap.
        plt.xticks(rotation=90)
        sns.set_context(font_scale=1)  # Increase font size.
        sns.heatmap(self.df.corr(), annot=nums, fmt=".1g", cmap='viridis', linecolor='white',
                    linewidth=linewidth)  # Draw heatmap.
        plt.savefig(f'./plots/heatmap/{filename}.png')  # Save heatmap result to location as png.
        plt.close()

    def get_violinplot(self):
        plt.figure(figsize=(16, 12))  # Increase figure size to fit the heatmap.
        sns.set_context(font_scale=1.8)  # Increase font size.
        sns.violinplot(data=self.df,
                       x='key',
                       y='valence',
                       hue='mode',
                       split=True, )
        plt.savefig('./plots/violin.png')  # Save heatmap result to location as png.
        plt.close()

    def get_histograms(self, features, rows, columns):
        fig = plt.figure(figsize=(20, 20))
        colors = sns.color_palette("brg", len(features))
        sns.set(style="darkgrid", font_scale=1.3)

        for i, feature in enumerate(features):
            if i == 0:
                pass # Skip index value
            else:
                ax = fig.add_subplot(rows, columns, i)
                sns.histplot(self.df[feature], bins=25, kde=True, ax=ax, color=colors[i])

        fig.tight_layout()
        plt.savefig('./plots/histograms.png')
        plt.close()

    def get_pairplot(self):
        sns.pairplot(self.df.sample(200, random_state=1), hue='mode')
        plt.savefig(f'./plots/pairplot_modes.png')
        plt.close()

    def get_pairplot_kde(self):
        grid = sns.PairGrid(self.df.sample(200, random_state=1))
        grid.map_offdiag(sns.kdeplot)
        grid.map_diag(plt.hist)
        plt.savefig(f'./plots/pairplot_kde.png')
        plt.close()

    def get_count_plot(self, feature, order=None):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=self.df, palette="gnuplot2", order=order)
        plt.title(f"Counts in each {feature}")
        plt.savefig(f'./plots/count/{feature}.png')
        plt.close()

    def get_box_plots(self, features):
        for i, feature in enumerate(features):
            fig = plt.figure(figsize=(20, 10))
            sns.set(style="darkgrid", font_scale=2)
            sns.boxplot(data=self.df, x='music_genre', y=feature)
            plt.title(f"Box Plot of: {feature}")
            plt.xticks(rotation=20)
            plt.savefig(f'./plots/box_plots/{feature}1.png')
            plt.close()

    def get_plots(self):
        self.get_heatmap("heatmap", nums=True, linewidth=1)
        self.get_pairplot()
        self.get_violinplot()
        self.get_histograms(self.numeric_features.columns, 5, 3)
        self.get_pairplot_kde()
        self.get_count_plot("key")
        self.get_count_plot("mode")
        self.get_count_plot("music_genre")
        self.get_box_plots(self.numeric_features.columns)
