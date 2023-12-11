import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import requests
from bs4 import BeautifulSoup
import warnings

class MovieAnalysis:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.scaler = MinMaxScaler()
        self.reg = RandomForestRegressor()

    def get_movie_features(self, urls):
        tab_indices = {
            0: [0],  # only cast box
            1: [0],  # only director crew
            2: [0],  # only studios
            3: [0]   # only genres
        }

        all_movie_data = []

        for url in urls:
            data = requests.get(url)
            soup = BeautifulSoup(data.text, 'html.parser')

            df_data = {}
            df_data['Title'] = soup.select('h1.headline-1')[0].text
            for tab, indices in tab_indices.items():
                try:
                    df_data[tab] = []
                    for index in indices:
                        for group in soup.select('div.tabbed-content-block')[tab].select('div.text-sluglist')[index].find_all('a'):
                            for g in group:
                                df_data[tab].append(g.text)
                except:
                    continue

            key_mapping = {
                0: "Cast",
                1: "Crew",
                2: 'Studios',
                3: 'Genres'
            }

            # Create a new dictionary with renamed keys
            new_dict = {key_mapping.get(key, key): value for key, value in df_data.items()}
            new_dict['URL'] = url
            all_movie_data.append(new_dict)

        df = pd.DataFrame(all_movie_data)
        return df

    def generate_list(self, df, feature_names): # Create a list of all unique feature values
        num_features = len(feature_names)

        result_lists = {}  # To store results for each feature

        for idx, feature_name in enumerate(feature_names):
            feature_dict = {}

            for index, row in df.iterrows():
                feats = row[feature_name]
                for sub_feat in feats:
                    if sub_feat not in feature_dict:
                        feature_dict[sub_feat] = (df['Rating'][index], 1)
                    else:
                        feature_dict[sub_feat] = (
                            feature_dict[sub_feat][0] + df['Rating'][index],
                            feature_dict[sub_feat][1] + 1
                        )

            for key in feature_dict: # Calculate average ratings for each feature
                feature_dict[key] = feature_dict[key][0] / feature_dict[key][1]

            lst = [(value, key) for key, value in feature_dict.items()] # Create and sort a list of tuples (dictionary value, key)
            lst = sorted(lst)

            feature_list = [element[1] for element in lst]
            ratings_list = [element[0] for element in lst]


            result_lists[feature_name] = feature_list

        return result_lists  

    def calculate_bin_array(self, this_list, all_features):
        return [1 if element in this_list else 0 for element in all_features]


    def split_arr(self, arr, n_splits):
        for i in range(0, len(arr), n_splits):
            yield arr[i:i + n_splits]

    def find_concentration(self, arr, n=3):
        batches = list(self.split_arr(arr, int(len(arr) / n)))
        concentrations = []
        for i in range(len(batches)):
            point = 0
            num_ones = 0
            for j in range(len(batches[i])):
                if batches[i][j] == 1:
                    point += j + (i * int(len(arr) / n))
                    num_ones += 1
            if num_ones > 0:
                point = point / num_ones
                concentrations.append((point, num_ones))
        return concentrations

    def to_concentrations(self, df, feature_names):
        for feature_name in feature_names:
            df[feature_name] = df[feature_name].apply(lambda x: self.find_concentration(x))
        return df

    def w_avg(self, arr):
        total_weight = sum(element[1] for element in arr)
        weighted_sum = sum(element[0] * element[1] for element in arr)

        return weighted_sum / total_weight if total_weight != 0 else 0


    def to_weighted_avg(self, df, feature_names):
        df[feature_names] = df[feature_names].applymap(self.w_avg)
        return df
    

    def train(self, df):
        df['Cast'] = df['Cast'].apply(json.loads)
        df['Crew'] = df['Crew'].apply(json.loads)
        df['Studios'] = df['Studios'].apply(json.loads)
        df['Genres'] = df['Genres'].apply(json.loads)
        
        categories = ['Genres', 'Crew', 'Cast', 'Studios']
        result_lists = self.generate_list(df, categories)
        self.genres_list = result_lists['Genres']
        self.crew_list = result_lists['Crew']
        self.cast_list = result_lists['Cast']
        self.studio_list = result_lists['Studios']
        self.df_bin_array = df.copy()
        
        self.df_bin_array['Cast'] = df['Cast'].apply(lambda x: self.calculate_bin_array(x, self.cast_list))
        self.df_bin_array['Crew'] = df['Crew'].apply(lambda x: self.calculate_bin_array(x, self.crew_list))
        self.df_bin_array['Studios'] = df['Studios'].apply(lambda x: self.calculate_bin_array(x, self.studio_list))
        self.df_bin_array['Genres'] = df['Genres'].apply(lambda x: self.calculate_bin_array(x, self.genres_list))
        self.df_bin_array = self.df_bin_array[['Title', 'Rating', 'Cast', 'Crew', 'Studios', 'Genres']]
        movies_shortened = self.to_concentrations(self.df_bin_array, ['Cast', 'Crew', 'Studios', 'Genres'])
        movies_shortened = self.to_weighted_avg(movies_shortened, ['Cast', 'Crew', 'Studios', 'Genres'])
        feat_df = movies_shortened[['Cast', 'Crew', 'Studios']]

        feat_scaled = pd.DataFrame(self.scaler.fit_transform(feat_df.astype(float)))
        feat_scaled.index = feat_df.index
        feat_scaled.columns = feat_df.columns

        target_df = pd.DataFrame()
        target_df['Rating'] = movies_shortened['Rating']

        X_train, X_test, y_train, y_test = train_test_split(feat_scaled, target_df, test_size = 0.2)
        
        self.reg.fit(X_train.values, y_train.values)
        target_pred = self.reg.predict(X_test.values)

        score = r2_score(y_test, target_pred)
        print("R^2 Score for predictions:", score)
        return self.reg

    def predict_new_movie(self, title):
        url = f'https://letterboxd.com/film/{title}/'
        new_row = self.get_movie_features([url]).drop(['Genres'], axis=1)
        self.df_bin_array = self.df_bin_array.append(new_row, ignore_index=True)
        self.df = self.df.append(new_row, ignore_index=True)

        for col, lst in {'Cast': self.cast_list, 'Crew': self.crew_list, 'Studios': self.studio_list}.items():
            iterate_all = False
            lst = lst.copy()
            for el in self.df_bin_array.iloc[-1][col]:
                if el not in lst:
                    iterate_all = True
                    lst.append(el)
            if iterate_all:
                self.df_bin_array[col] = self.df[col].copy()
                self.df_bin_array[col] = self.df[col].apply(lambda x: self.calculate_bin_array(x, lst))
            else:
                self.df_bin_array.at[self.df_bin_array.shape[0] - 1, col] = self.df_bin_array.iloc[-1:][col].apply(
                    lambda x: self.calculate_bin_array(x, lst)).copy().values[0]

        if iterate_all:
            movies_shortened = self.to_concentrations(self.df_bin_array, ['Cast', 'Crew', 'Studios'])
            movies_shortened = self.to_weighted_avg(movies_shortened, ['Cast', 'Crew', 'Studios'])
        else:
            movies_shortened = self.to_concentrations(self.df_bin_array.iloc[-1:], ['Cast', 'Crew', 'Studios'])
            movies_shortened = self.to_weighted_avg(movies_shortened.iloc[-1:], ['Cast', 'Crew', 'Studios'])

        feat_df = movies_shortened[['Cast', 'Crew', 'Studios']]

        scaled_new_row = self.scaler.transform([feat_df.iloc[-1]])
        scaled_new_df = pd.DataFrame(scaled_new_row, columns=feat_df.columns)
        pred_score = self.reg.predict(scaled_new_df.values)
        print(f'Predicted Score for {self.df.iloc[-1].Title} is: {pred_score}')

if __name__ == "__main__":
    movie_analysis = MovieAnalysis()
    model = movie_analysis.train('your_input_file.csv')
    movie_analysis.predict_new_movie('your_movie_title')