# ------------------------------------------------------------------
# -----------------------------LIBRARIES----------------------------
# ------------------------------------------------------------------

import os
import sys
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.helper import compute_person_stats, add_person_features, load_data


# ------------------------------------------------------------------
# -----------------------PREPROCESSING PIPELINE---------------------
# ------------------------------------------------------------------

def preprocess_pipeline(data_path: str) -> pd.DataFrame:
    """
    Preprocess IMDb data and compute features for movie risk assessment.

    Steps:
    1. Load raw IMDb data (titles, ratings, crew, principals, name_basic, akas)
    2. Filter movies only, remove adult films
    3. Merge rating information
    4. Process crew (directors, writers) and cast (top 10 actors/actresses)
    5. Lookup person names
    6. Compute person-based stats (mean rating, total films) for directors and cast
    7. Fill missing values
    8. Create target labels: is_success, is_risky

    Parameters
    ----------
    data_path : str
        Path to directory containing IMDb TSV files.

    Returns
    -------
    pd.DataFrame
        Preprocessed movies DataFrame with features ready for ML.
    """
    print("Starting data preprocessing pipeline...")
    
    # Create progress bar for main steps
    pbar = tqdm(total=7, desc="Preprocessing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # --- LOAD DATA ---
    pbar.set_description("Loading raw data")
    titles, ratings, crew, principals, name_basic, akas = load_data(data_path)
    pbar.update(1)

    # --- MOVIES ---
    """
    Filter only movies and remove adult films:
    - titleType=='movie' ensures we only consider movies.
    - Convert startYear and runtimeMinutes to numeric.
    - isAdult==0 filters out adult movies.
    - Add averageRating and numVotes to movies.
    - Missing ratings will appear as NaN.
    """
    pbar.set_description("Filtering movies")
    movies = titles.query("titleType=='movie'").copy()
    movies['startYear'] = pd.to_numeric(movies['startYear'], errors='coerce')
    movies['runtimeMinutes'] = pd.to_numeric(movies['runtimeMinutes'], errors='coerce')
    movies['isAdult'] = pd.to_numeric(movies['isAdult'], errors='coerce').fillna(0).astype(int)
    movies = movies[movies['isAdult']==0]
    movies = movies.merge(ratings[['tconst','averageRating','numVotes']], on='tconst', how='left')
    pbar.update(1)

    # --- CREW ---
    """
    Process directors and writers:
    - Replace '\\N' with NaN.
    - Drop movies with missing directors or writers.
    - Split comma-separated strings into lists.
    - Ensure each movie has at least one director and writer.
    - Merge directors and writers info into movies dataframe.
    """
    pbar.set_description("Processing crew (directors/writers)")
    crew['directors'] = crew['directors'].replace("\\N", pd.NA)
    crew['writers'] = crew['writers'].replace("\\N", pd.NA)
    crew = crew.dropna(subset=['directors','writers'])
    crew['directors_nconst'] = crew['directors'].apply(lambda s: [x for x in str(s).split(',') if x])
    crew['writers_nconst'] = crew['writers'].apply(lambda s: [x for x in str(s).split(',') if x])
    crew = crew[crew['directors_nconst'].apply(len)>0]
    crew = crew[crew['writers_nconst'].apply(len)>0]
    movies = movies.merge(crew[['tconst','directors_nconst','writers_nconst']], on='tconst', how='inner')
    pbar.update(1)

    # --- CAST ---
    """
    Process cast (actors/actresses):
    - Filter only actors and actresses.
    - Drop missing nconst values.
    - Sort by ordering (IMDb primary cast first).
    - Take top 10 cast members for each movie.
    - Aggregate cast nconsts into a list per movie.
    - Merge with movies dataframe.
    """
    pbar.set_description("Processing cast")
    cast = principals.query("category in ['actor','actress']").copy()
    cast = cast[cast['nconst'].notna()]
    cast = cast.sort_values(['tconst','ordering'])
    top_cast = cast.groupby('tconst').head(10)
    df_casts_groups = top_cast.groupby('tconst')['nconst'].apply(list).reset_index()
    df_casts_groups.rename(columns={'nconst':'cast_nconst'}, inplace=True)
    movies = movies.merge(df_casts_groups, on='tconst', how='inner')
    pbar.update(1)

    # --- NAME LOOKUP ---
    """
    Map nconst to person names for readability:
    - director_names: list of director names per movie
    - cast_names: list of cast names per movie
    """
    pbar.set_description("Mapping person names")
    nconst_name = dict(zip(name_basic['nconst'], name_basic['primaryName']))
    movies['director_names'] = movies['directors_nconst'].apply(lambda lst: [nconst_name.get(x) for x in lst])
    movies['cast_names'] = movies['cast_nconst'].apply(lambda lst: [nconst_name.get(x) for x in lst])
    pbar.update(1)

    # --- PERSON STATS ---
    """
    Compute track record stats:
    - Combine principal cast + directors + writers
    - Merge with ratings to get each person's averageRating
    - Compute mean rating and total films per person
    - Apply to directors, writers, and cast to get director_mean_rating, writer_mean_rating, etc.
    """
    pbar.set_description("Computing person statistics")
    person_movies = principals[['nconst','tconst']].copy()
    dir_expl = crew[['tconst','directors_nconst']].explode('directors_nconst').rename(columns={'directors_nconst':'nconst'})
    writer_expl = crew[['tconst','writers_nconst']].explode('writers_nconst').rename(columns={'writers_nconst':'nconst'})
    person_movies = pd.concat([person_movies, dir_expl[['nconst','tconst']], writer_expl[['nconst','tconst']]], ignore_index=True)
    mean_map, count_map = compute_person_stats(person_movies, ratings)
    movies = add_person_features(movies, mean_map, count_map, 'directors_nconst', 'director')
    movies = add_person_features(movies, mean_map, count_map, 'writers_nconst', 'writer')
    movies = add_person_features(movies, mean_map, count_map, 'cast_nconst', 'cast')
    pbar.update(1)

    # --- FILL NA ---
    """
    Fill remaining missing values for mean ratings and total films:
    - Use global mean for missing director/writer/cast mean ratings
    - Use 0 for missing total films
    """
    pbar.set_description("Filling missing values & creating labels")
    
    # Drop movies with missing critical fields
    # - startYear: Critical temporal feature - imputation would introduce bias
    # - averageRating/numVotes: Target variables for regression - cannot be missing
    movies = movies.dropna(subset=['startYear', 'averageRating', 'numVotes'])
    
    # Fill missing values for person stats
    global_director_mean = movies['director_mean_rating'].mean()
    global_writer_mean = movies['writer_mean_rating'].mean()
    global_cast_mean = movies['cast_mean_rating'].mean()
    movies['director_mean_rating'] = movies['director_mean_rating'].fillna(global_director_mean)
    movies['writer_mean_rating'] = movies['writer_mean_rating'].fillna(global_writer_mean)
    movies['cast_mean_rating'] = movies['cast_mean_rating'].fillna(global_cast_mean)
    movies['director_total_films'] = movies['director_total_films'].fillna(0)
    movies['writer_total_films'] = movies['writer_total_films'].fillna(0)
    movies['cast_total_films'] = movies['cast_total_films'].fillna(0)
    
    # Fill runtimeMinutes with median (missing runtime is less critical than missing year)
    # Typical movie runtime is fairly predictable from median
    movies['runtimeMinutes'] = movies['runtimeMinutes'].fillna(movies['runtimeMinutes'].median())

    # --- LABEL ---
    """
    Create target labels for ML:
    - is_success: 1 if averageRating>=7.0 AND numVotes>=30000, else 0
    - is_risky: 1 - is_success (for investment risk analysis)
    """
    movies['is_success'] = ((movies['averageRating']>=7.0) & (movies['numVotes']>=30000)).astype(int)
    movies['is_risky'] = 1 - movies['is_success']

    # --- GENRES ---
    """
    Parse genres column into list format:
    - Split comma-separated genre strings into lists
    - Handle missing values
    """
    movies['genre_list'] = movies['genres'].apply(
        lambda x: [g.strip() for g in str(x).split(',') if g.strip() and g.strip() != '\\N'] if pd.notna(x) else []
    )
    pbar.update(1)
    
    pbar.close()
    print(f"Preprocessing complete! Processed {len(movies)} movies.")
    return movies


