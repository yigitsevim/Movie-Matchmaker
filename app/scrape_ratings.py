import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

def rating_to_numeric(film_rating_raw):
    stars = film_rating_raw.count('★')
    halves = film_rating_raw.count('½')
    rating_numeric = stars + halves/2
    return rating_numeric

def get_movie_features(urls):
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

def scrape(username):
    url = f'https://letterboxd.com/{username}/films/page/0/'
    data = requests.get(url)
    soup = BeautifulSoup(data.text, 'html.parser')
    try:
        pages = soup.select('div.pagination')[0]
        pages = pages.find_all('a')
        n_pages = pages[-1:][0].text
    except IndexError:
        n_pages = 1

    print(f'{n_pages} pages of rated movies found')

    film_url_list = []
    film_name_list = []
    film_rating_list = []
    for page_idx in range(1, int(n_pages)+1):
        url = f'https://letterboxd.com/{username}/films/page/{page_idx}/'
        data = requests.get(url)
        soup = BeautifulSoup(data.text, 'html.parser')
        posters = soup.select('li.poster-container')

        for poster in posters:
            try:
                film_url = poster.find('div', class_='film-poster')['data-film-slug']
                film_url = f'https://letterboxd.com/film/{film_url}'
                film_name = poster.find('img', class_='image')['alt']
                film_rating_raw = poster.find('span', class_='rating').text
                film_rating_numeric = rating_to_numeric(film_rating_raw)

                film_url_list.append(film_url)
                film_name_list.append(film_name)
                film_rating_list.append(film_rating_numeric)
            except:
                continue
    df_data = {
        'Title': film_name_list,
        'Rating': film_rating_list,
        'URL': film_url_list
    }
    rating_df = pd.DataFrame(data=df_data)
    movie_features = get_movie_features(rating_df.URL.values)
    df = rating_df.merge(movie_features, on=['Title', 'URL'])

    df['Cast'] = df['Cast'].apply(json.dumps)
    df['Crew'] = df['Crew'].apply(json.dumps)
    df['Studios'] = df['Studios'].apply(json.dumps)
    df['Genres'] = df['Genres'].apply(json.dumps)

    return df