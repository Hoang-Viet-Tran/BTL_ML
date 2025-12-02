# ------------------------------------------------------------------
# -----------------------------LIBRARIES----------------------------
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union

# ------------------------------------------------------------------
# -----------------------------LOAD DATA----------------------------
# ------------------------------------------------------------------

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required IMDb TSV files from the given directory.

    Parameters
    ----------
    data_path : str
        Path to the directory containing IMDb TSV files.

    Returns
    -------
    titles : pd.DataFrame
        DataFrame containing title basics.
    ratings : pd.DataFrame
        DataFrame containing title ratings.
    crew : pd.DataFrame
        DataFrame containing directors and writers for each title.
    principals : pd.DataFrame
        DataFrame containing the cast and crew of titles.
    name_basic : pd.DataFrame
        DataFrame containing basic information of people (actors, directors, etc.).
    akas : pd.DataFrame
        DataFrame containing title alternate names.
    """
    titles = pd.read_csv(f"{data_path}/title.basics.tsv", sep='\t')
    ratings = pd.read_csv(f"{data_path}/title.ratings.tsv", sep='\t')
    crew = pd.read_csv(f"{data_path}/title.crew.tsv", sep='\t')
    principals = pd.read_csv(f"{data_path}/title.principals.tsv", sep='\t')
    name_basic = pd.read_csv(f"{data_path}/name.basics.tsv", sep='\t')
    akas = pd.read_csv(f"{data_path}/title.akas.tsv", sep='\t')
    return titles, ratings, crew, principals, name_basic, akas

# ------------------------------------------------------------------
# -----------------------AGGREGATE MEAN/COUNT-----------------------
# ------------------------------------------------------------------

def agg_person_metric(nconst_list: Union[List[str], None], metric_map: Dict[str, float], func=np.mean) -> float:
    """
    Aggregate a list of person IDs (nconst) using a provided metric map and aggregation function.

    Parameters
    ----------
    nconst_list : list of str or None
        List of person IDs (e.g., directors or actors). Can be None or not a list.
    metric_map : dict
        Mapping from person ID to a numeric metric (e.g., mean rating or total films).
    func : callable, default=np.mean
        Aggregation function to apply to the metric values (mean, sum, etc.).

    Returns
    -------
    float
        Aggregated metric for the given list of person IDs. Returns np.nan if the list is invalid or empty.
    """
    if not isinstance(nconst_list, list):
        return np.nan
    vals = [metric_map.get(x) for x in nconst_list if metric_map.get(x) is not None]
    return func(vals) if len(vals) else np.nan

# ------------------------------------------------------------------
# ---------------------------COMPUTE STATS--------------------------
# ------------------------------------------------------------------

def compute_person_stats(person_movies: pd.DataFrame, ratings: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Compute aggregated statistics (mean rating and total films) for each person (actor, director, etc.).

    Parameters
    ----------
    person_movies : pd.DataFrame
        DataFrame containing columns ['nconst', 'tconst'] mapping people to titles.
    ratings : pd.DataFrame
        DataFrame containing ['tconst', 'averageRating', 'numVotes'].

    Returns
    -------
    mean_map : dict
        Mapping from person ID (nconst) to their mean rating across all titles.
    count_map : dict
        Mapping from person ID (nconst) to the number of titles they participated in.
    """

    person_movies = person_movies.merge(
        ratings[['tconst','averageRating','numVotes']],
        on='tconst', how='left'
    )
    person_stats = person_movies.groupby('nconst').agg(
        person_mean_rating=('averageRating','mean'),
        person_count_films=('tconst','nunique')
    ).reset_index()
    mean_map = dict(zip(person_stats['nconst'], person_stats['person_mean_rating']))
    count_map = dict(zip(person_stats['nconst'], person_stats['person_count_films']))
    return mean_map, count_map

def add_person_features(df: pd.DataFrame, mean_map: Dict[str, float], count_map: Dict[str, int], col_list: str, prefix: str) -> pd.DataFrame:
    """
    Add aggregated person-based features to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a column of person IDs (list of nconsts).
    mean_map : dict
        Mapping from person ID to mean rating.
    count_map : dict
        Mapping from person ID to total number of films.
    col_list : str
        Column name in df that contains list of person IDs.
    prefix : str
        Prefix to use for the new columns (e.g., 'director', 'cast').

    Returns
    -------
    pd.DataFrame
        DataFrame with two new columns:
        - {prefix}_mean_rating : aggregated mean rating of listed persons.
        - {prefix}_total_films : aggregated total films of listed persons.
    """
    df[f'{prefix}_mean_rating'] = df[col_list].apply(lambda L: agg_person_metric(L, mean_map))
    df[f'{prefix}_total_films'] = df[col_list].apply(lambda L: agg_person_metric(L, count_map, func=np.sum))
    return df

