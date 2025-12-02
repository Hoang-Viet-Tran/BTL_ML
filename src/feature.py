import pandas as pd
import numpy as np
from typing import Tuple

# ------------------------------------------------------------------
# ------------------------------GENRE ENCODED-----------------------
# ------------------------------------------------------------------

def encode_genres(df: pd.DataFrame, genre_col: str = 'genre_list') -> pd.DataFrame:
    """
    Convert a column containing lists of movie genres into multiple binary (0/1) columns, 
    one for each unique genre in the dataset. This is a form of multi-hot encoding 
    suitable for machine learning models.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing a column of genre lists.
    genre_col : str, default 'genre_list'
        The name of the column in df which contains lists of genres.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with additional columns, one for each unique genre.
        Each new column has value 1 if the movie belongs to that genre, 0 otherwise.

    Example
    -------
    df['genre_list'] = [['Action','Adventure'], ['Drama'], ['Action','Drama']]
    df = encode_genres(df)
    # Creates columns: genre_Action, genre_Adventure, genre_Drama
    # Resulting DataFrame:
    #   genre_list               genre_Action  genre_Adventure  genre_Drama
    # 0 ['Action','Adventure']   1             1               0
    # 1 ['Drama']                0             0               1
    # 2 ['Action','Drama']       1             0               1
    """
    all_genres = set(g for sublist in df[genre_col] for g in sublist if isinstance(sublist, list))
    for g in all_genres:
        df[f'genre_{g}'] = df[genre_col].apply(lambda x: int(g in x) if isinstance(x, list) else 0)
    return df

# ------------------------------------------------------------------
# --------------------------------FEATURING-------------------------
# ------------------------------------------------------------------

def get_feature_target(df: pd.DataFrame, target_type: str = 'rating') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into feature matrix X and target vector/matrix y for ML models.
    
    Enhanced version with support for multiple target types and writer features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after feature engineering, including person stats and genre encoding.
    target_type : str, default 'rating'
        Type of target variable:
        - 'rating': averageRating (single-output regression)
        - 'votes': numVotes (single-output regression)
        - 'rating_votes': both averageRating and numVotes (multi-output regression)
        - 'is_risky': binary classification target

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing:
        - Director stats: director_mean_rating, director_total_films
        - Writer stats: writer_mean_rating, writer_total_films
        - Cast stats: cast_mean_rating, cast_total_films
        - Genre multi-hot encoded columns: genre_Action, genre_Drama, etc.
        - Numeric movie info: runtimeMinutes, startYear, director_year_gap (if available)
    y : pd.Series or pd.DataFrame
        Target variable(s). Single column for single-output tasks, 
        multiple columns for multi-output regression.

    Notes
    -----
    - Automatically detects available feature columns
    - Backwards compatible with old target_col parameter via target_type='rating'
    """
    # Extract all relevant numeric feature columns
    # Exclude columns containing lists (directors_nconst, writers_nconst, cast_nconst, genre_list, etc.)
    exclude_cols = ['directors_nconst', 'writers_nconst', 'cast_nconst', 'genre_list', 
                    'director_names', 'cast_names', 'tconst', 'primaryTitle', 
                    'titleType', 'originalTitle', 'isAdult', 'endYear', 'genres',
                    'averageRating', 'numVotes', 'is_success', 'is_risky',
                    'directors', 'writers', 'ordering', 'nconst', 'category']
    
    feature_cols = []
    for c in df.columns:
        # Include specific stat columns
        if (c.endswith('_mean_rating') or c.endswith('_total_films') or 
            c.startswith('genre_') or c in ['runtimeMinutes', 'startYear', 'director_year_gap']):
            if c not in exclude_cols:
                feature_cols.append(c)
    
    X = df[feature_cols]
    
    # Handle different target types
    if target_type == 'rating':
        y = df['averageRating']
    elif target_type == 'votes':
        y = df['numVotes']
    elif target_type == 'rating_votes':
        y = df[['averageRating', 'numVotes']]
    elif target_type == 'is_risky':
        y = df['is_risky']
    else:
        # Backwards compatibility: treat as column name
        y = df[target_type] if target_type in df.columns else df['averageRating']
    
    return X, y
