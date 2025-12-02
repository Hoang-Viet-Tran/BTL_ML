# ------------------------------------------------------------------
# -----------------------------LIBRARIES----------------------------
# ------------------------------------------------------------------
import os
import json
import math
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    roc_auc_score
)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from xgboost import XGBRegressor

from preprocess import preprocess_pipeline
from feature import encode_genres, get_feature_target


# ------------------------------------------------------------------
# ---------------------TRAIN CLASSIFICATION MODELS------------------
# ------------------------------------------------------------------

def train_is_risky_models(
    data_path: str,
    save_dir: str = "models/is_risky"
) -> Dict[str, Any]:
    """
    Train classification models for Y = is_risky.

    Models:
        - Logistic Regression
        - Random Forest Classifier

    Steps:
        1. Load data
        2. Preprocess pipeline
        3. Feature engineering (genres encoding, others)
        4. Extract X, y via get_feature_target()
        5. Train/test split
        6. Train 2 models (with light tuning)
        7. Save best models
        8. Return model dict

    Returns:
        dict: {
            "logreg": trained_logistic_model,
            "rf": trained_random_forest_model
        }
    """
     # --------------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------------
    df = pd.read_csv(data_path)

    # --------------------------------------------------------------
    # 2. Preprocess (clean missing, normalize fields…)
    # --------------------------------------------------------------
    df = preprocess_pipeline(df)

    # --------------------------------------------------------------
    # 3. Feature engineering (genres one-hot, etc.)
    # --------------------------------------------------------------
    df = encode_genres(df)

    # --------------------------------------------------------------
    # 4. Extract X, y for target = "is_risky"
    # --------------------------------------------------------------
    X, y = get_feature_target(df, target_type="is_risky")

    # --------------------------------------------------------------
    # 5. Train/test split
    # --------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------------
    # 6. Train Logistic Regression
    # --------------------------------------------------------------
    logreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    logreg_param_grid = {
        "clf__C": [0.1, 1.0, 5.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }

    logreg_grid = GridSearchCV(
        logreg_pipeline,
        param_grid=logreg_param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    logreg_grid.fit(X_train, y_train)
    best_logreg = logreg_grid.best_estimator_

    # --------------------------------------------------------------
    # 7. Train Random Forest
    # --------------------------------------------------------------
    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier())
    ])

    rf_param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        param_grid=rf_param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # --------------------------------------------------------------
    # 8. Save models
    # --------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(best_logreg, f"{save_dir}/logreg_is_risky.pkl")
    joblib.dump(best_rf, f"{save_dir}/rf_is_risky.pkl")

    # --------------------------------------------------------------
    # 9. Return dictionary of trained models
    # --------------------------------------------------------------
    return {
        "logreg": best_logreg,
        "rf": best_rf,
    }

# ------------------------------------------------------------------
# --------------------------TRAIN REGRESSION------------------------
# ------------------------------------------------------------------

def train_rating_votes_models(
    data_path: str,
    save_dir: str = "models/rating_votes"
) -> Dict[str, Any]:
    """
    Train regression models for:
        - averageRating
        - numVotes
    (Supports multi-output if implemented in get_feature_target)

    Models:
        - RandomForestRegressor
        - XGBRegressor

    Returns:
        dict: {
            "rf_reg": best_rf,
            "xgb_reg": best_xgb
        }
    """

    # --------------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------------
    df = pd.read_csv(data_path)

    # --------------------------------------------------------------
    # 2. Preprocess
    # --------------------------------------------------------------
    df = preprocess_pipeline(df)

    # --------------------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------------------
    df = encode_genres(df)

    # --------------------------------------------------------------
    # 4. Extract X, y for regression
    # --------------------------------------------------------------
    X, y = get_feature_target(df, target_type="rating_votes")

    # --------------------------------------------------------------
    # 5. Train/test split
    # --------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==============================================================
    # --------------------- MODEL 1: RANDOM FOREST ------------------
    # ==============================================================

    rf_pipeline = Pipeline([
        ("clf", RandomForestRegressor())
    ])

    rf_param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        param_grid=rf_param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    # Multi-output RF (if y is multi-output)
    if len(y.shape) == 2:
        rf_model = MultiOutputRegressor(rf_grid)
    else:
        rf_model = rf_grid

    rf_model.fit(X_train, y_train)

    # Extract best model (for multi-output)
    if isinstance(rf_model, MultiOutputRegressor):
        best_rf = rf_model
    else:
        best_rf = rf_grid.best_estimator_

    # Evaluate (optional)
    rf_pred = best_rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    print(f"[RandomForest] MAE = {rf_mae:.4f}")


    # ==============================================================
    # ------------------------ MODEL 2: XGBOOST ---------------------
    # ==============================================================

    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    # Multi-output XGB (wrap)
    if len(y.shape) == 2:
        xgb_model = MultiOutputRegressor(xgb)
    else:
        xgb_model = xgb

    xgb_model.fit(X_train, y_train)

    # Evaluate
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    print(f"[XGBoost] MAE = {xgb_mae:.4f}")


    # --------------------------------------------------------------
    # 6. Save models
    # --------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(best_rf, f"{save_dir}/rf_rating_votes.pkl")
    joblib.dump(xgb_model, f"{save_dir}/xgb_rating_votes.pkl")

    # --------------------------------------------------------------
    # 7. Return dictionary
    # --------------------------------------------------------------
    return {
        "rf_reg": best_rf,
        "xgb_reg": xgb_model,
    }


# ------------------------------------------------------------------
# --------------------------TRAIN HYBRID----------------------------
#  Head classification (is_risky), Head regression (rating + votes)
# ------------------------------------------------------------------

# ---------------------------
# UTILS: seed, device
# ---------------------------
RND_SEED = 42
pl.seed_everything(RND_SEED)
torch.backends.cudnn.benchmark = True


# ---------------------------
# VOCAB BUILDING HELPERS
# ---------------------------
def build_topk_vocab(list_of_lists: List[List[str]], topk: int, unk_token: str = "<UNK>") -> Dict[str, int]:
    """
    Count frequency of tokens in list_of_lists and return mapping token->idx for topk tokens.
    """
    from collections import Counter
    cnt = Counter()
    for lst in list_of_lists:
        if isinstance(lst, list):
            cnt.update(lst)
    most_common = cnt.most_common(topk)
    vocab = {unk_token: 0}
    idx = 1
    for tok, _ in most_common:
        vocab[tok] = idx
        idx += 1
    return vocab


def tokens_to_indices(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """
    Map token list to indices using vocab; unknown -> 0 index.
    """
    if not isinstance(tokens, list):
        return []
    return [vocab.get(t, 0) for t in tokens]


# ---------------------------
# DATASET
# ---------------------------
class IMDBHybridDataset(Dataset):
    """
    Dataset expects a preprocessed DataFrame `df` with columns:
      - directors_nconst (list[str])
      - cast_nconst (list[str])
      - writers_nconst (list[str])  # if available
      - genre_list (list[str])      # after preprocess
      - numeric features (director_mean_rating, director_total_films, cast_mean_rating, cast_total_films, runtimeMinutes, etc.)
      - averageRating (float), numVotes (int), is_risky (0/1)
    We will convert lists to indices using supplied vocabs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dir_vocab: Dict[str, int],
        cast_vocab: Dict[str, int],
        writer_vocab: Dict[str, int],
        numeric_cols: List[str],
        genre_cols: List[str],
        max_cast_len: int = 10,
        max_dir_len: int = 3,
        max_writer_len: int = 5,
    ):
        self.df = df.reset_index(drop=True)
        self.dir_vocab = dir_vocab
        self.cast_vocab = cast_vocab
        self.writer_vocab = writer_vocab
        self.numeric_cols = numeric_cols
        self.genre_cols = genre_cols

        self.max_cast_len = max_cast_len
        self.max_dir_len = max_dir_len
        self.max_writer_len = max_writer_len

        # Precompute mapped columns to speed up __getitem__
        self._map_person_cols()

    def _map_person_cols(self):
        # produce token index lists for each record
        self.dir_idx = []
        self.cast_idx = []
        self.writer_idx = []
        for _, row in self.df.iterrows():
            dlist = tokens_to_indices(row.get('directors_nconst', []), self.dir_vocab)
            clist = tokens_to_indices(row.get('cast_nconst', []), self.cast_vocab)
            wlist = tokens_to_indices(row.get('writers_nconst', []), self.writer_vocab) if 'writers_nconst' in row else []
            # truncate to max lengths (we will pad in collate)
            self.dir_idx.append(dlist[:self.max_dir_len])
            self.cast_idx.append(clist[:self.max_cast_len])
            self.writer_idx.append(wlist[:self.max_writer_len])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # numeric features vector
        numeric_vector = row[self.numeric_cols].fillna(0).to_numpy(dtype=np.float32)
        # genres vector (already one-hot columns)
        genre_vector = row[self.genre_cols].to_numpy(dtype=np.float32)
        # person lists (indices)
        d = torch.tensor(self.dir_idx[idx], dtype=torch.long)
        c = torch.tensor(self.cast_idx[idx], dtype=torch.long)
        w = torch.tensor(self.writer_idx[idx], dtype=torch.long) if self.writer_idx[idx] is not None else torch.tensor([], dtype=torch.long)
        # targets
        is_risky = torch.tensor(row['is_risky'], dtype=torch.float32)
        avg_rating = torch.tensor(row['averageRating'] if not pd.isna(row['averageRating']) else 0.0, dtype=torch.float32)
        num_votes = torch.tensor(row['numVotes'] if not pd.isna(row['numVotes']) else 0.0, dtype=torch.float32)
        return {
            'numeric': torch.from_numpy(numeric_vector),
            'genre': torch.from_numpy(genre_vector),
            'dirs': d,
            'cast': c,
            'writers': w,
            'is_risky': is_risky,
            'avg_rating': avg_rating,
            'num_votes': num_votes
        }


# ---------------------------
# COLLATE FN
# ---------------------------
def hybrid_collate_fn(batch: List[Dict[str, Any]]):
    """
    Collate list of samples into batched tensors.
    - Pads dirs/cast/writers lists to max length in batch (right padding).
    - Returns dict of batched tensors.
    """
    numeric = torch.stack([b['numeric'] for b in batch])           # (B, num_numeric)
    genre = torch.stack([b['genre'] for b in batch])               # (B, num_genres)
    is_risky = torch.stack([b['is_risky'] for b in batch]).unsqueeze(1)  # (B,1)
    avg_rating = torch.stack([b['avg_rating'] for b in batch]).unsqueeze(1)
    num_votes = torch.stack([b['num_votes'] for b in batch]).unsqueeze(1)

    # pad sequences
    dirs_seq = [b['dirs'] for b in batch]
    cast_seq = [b['cast'] for b in batch]
    writers_seq = [b['writers'] for b in batch]

    dirs_padded = pad_sequence(dirs_seq, batch_first=True, padding_value=0) if len(dirs_seq) and dirs_seq[0].numel() != 0 else torch.zeros((len(batch), 0), dtype=torch.long)
    cast_padded = pad_sequence(cast_seq, batch_first=True, padding_value=0) if len(cast_seq) and cast_seq[0].numel() != 0 else torch.zeros((len(batch), 0), dtype=torch.long)
    writers_padded = pad_sequence(writers_seq, batch_first=True, padding_value=0) if len(writers_seq) and writers_seq[0].numel() != 0 else torch.zeros((len(batch), 0), dtype=torch.long)

    return {
        'numeric': numeric,
        'genre': genre,
        'dirs': dirs_padded,
        'cast': cast_padded,
        'writers': writers_padded,
        'is_risky': is_risky,
        'avg_rating': avg_rating,
        'num_votes': num_votes
    }


# ---------------------------
# MODEL 1: Simpler DNN (No Embeddings)
# ---------------------------
class SimplerDNN(pl.LightningModule):
    """
    Simpler Deep Neural Network for multi-task learning.
    Uses only numeric and genre features (no person embeddings).
    Faster to train and good baseline for comparison with HybridMovieModel.
    """
    def __init__(
        self,
        numeric_dim: int,
        genre_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cls_weight: float = 1.0,
        rating_weight: float = 1.0,
        votes_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Input batch normalization
        self.numeric_bn = nn.BatchNorm1d(numeric_dim) if numeric_dim > 0 else nn.Identity()
        self.genre_bn = nn.BatchNorm1d(genre_dim) if genre_dim > 0 else nn.Identity()
        
        # Build shared MLP backbone
        input_dim = numeric_dim + genre_dim
        layers = []
        cur_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(cur_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            cur_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        
        # Task-specific heads
        self.cls_head = nn.Linear(cur_dim, 1)  # Classification
        self.rating_head = nn.Linear(cur_dim, 1)  # Rating regression
        self.votes_head = nn.Linear(cur_dim, 1)  # Votes regression (log-scaled)
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.cls_weight = cls_weight
        self.rating_weight = rating_weight
        self.votes_weight = votes_weight
    
    def forward(self, numeric, genre):
        """
        Forward pass using only numeric and genre features.
        
        Args:
            numeric: (B, numeric_dim)
            genre: (B, genre_dim)
        
        Returns:
            cls_logits, rating_out, votes_out
        """
        # Normalize inputs
        numeric = self.numeric_bn(numeric)
        genre = self.genre_bn(genre)
        
        # Concatenate features
        x = torch.cat([numeric, genre], dim=1)
        
        # Shared backbone
        features = self.backbone(x)
        
        # Task-specific predictions
        cls_logits = self.cls_head(features).squeeze(1)
        rating_out = self.rating_head(features).squeeze(1)
        votes_out = self.votes_head(features).squeeze(1)
        
        return cls_logits, rating_out, votes_out
    
    def training_step(self, batch, batch_idx):
        numeric = batch['numeric'].float()
        genre = batch['genre'].float()
        
        cls_logits, rating_pred, votes_pred = self.forward(numeric, genre)
        
        # Targets
        is_risky = batch['is_risky'].squeeze(1)
        avg_rating = batch['avg_rating'].squeeze(1)
        num_votes = batch['num_votes'].squeeze(1)
        log_votes = torch.log1p(num_votes)
        
        # Losses
        loss_cls = self.bce_loss(cls_logits, is_risky)
        loss_rating = self.l1_loss(rating_pred, avg_rating)
        loss_votes = self.l1_loss(votes_pred, log_votes)
        
        loss = self.cls_weight * loss_cls + self.rating_weight * loss_rating + self.votes_weight * loss_votes
        
        # Logging
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/loss_cls', loss_cls, on_epoch=True)
        self.log('train/loss_rating', loss_rating, on_epoch=True)
        self.log('train/loss_votes', loss_votes, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        numeric = batch['numeric'].float()
        genre = batch['genre'].float()
        
        cls_logits, rating_pred, votes_pred = self.forward(numeric, genre)
        
        is_risky = batch['is_risky'].squeeze(1)
        avg_rating = batch['avg_rating'].squeeze(1)
        num_votes = batch['num_votes'].squeeze(1)
        log_votes = torch.log1p(num_votes)
        
        loss_cls = self.bce_loss(cls_logits, is_risky)
        loss_rating = self.l1_loss(rating_pred, avg_rating)
        loss_votes = self.l1_loss(votes_pred, log_votes)
        loss = self.cls_weight * loss_cls + self.rating_weight * loss_rating + self.votes_weight * loss_votes
        
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        
        return {
            'loss': loss.detach(),
            'preds_cls': torch.sigmoid(cls_logits).detach().cpu().numpy(),
            'targets_cls': is_risky.detach().cpu().numpy(),
            'preds_rating': rating_pred.detach().cpu().numpy(),
            'targets_rating': avg_rating.detach().cpu().numpy(),
            'preds_votes': votes_pred.detach().cpu().numpy(),
            'targets_votes': log_votes.detach().cpu().numpy()
        }
    
    def validation_epoch_end(self, outputs):
        # Collect predictions
        preds_cls = np.concatenate([o['preds_cls'] for o in outputs])
        targets_cls = np.concatenate([o['targets_cls'] for o in outputs])
        preds_rating = np.concatenate([o['preds_rating'] for o in outputs])
        targets_rating = np.concatenate([o['targets_rating'] for o in outputs])
        
        # Metrics
        try:
            auc = roc_auc_score(targets_cls, preds_cls)
        except Exception:
            auc = 0.0
        preds_bin = (preds_cls >= 0.5).astype(int)
        f1 = f1_score(targets_cls, preds_bin, zero_division=0)
        mae_rating = mean_absolute_error(targets_rating, preds_rating)
        
        self.log('val/auc', auc, prog_bar=True)
        self.log('val/f1', f1)
        self.log('val/mae_rating', mae_rating)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


# ---------------------------
# MODEL 2: Hybrid Multi-task (with Embeddings)
# ---------------------------
class HybridMovieModel(pl.LightningModule):
    def __init__(
        self,
        numeric_dim: int,
        genre_dim: int,
        dir_vocab_size: int,
        cast_vocab_size: int,
        writer_vocab_size: int,
        dir_emb_dim: int = 32,
        cast_emb_dim: int = 64,
        writer_emb_dim: int = 32,
        shared_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        cls_weight: float = 1.0,
        rating_weight: float = 1.0,
        votes_weight: float = 1.0,
    ):
        super().__init__()
        # save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Embeddings
        self.dir_emb = nn.Embedding(num_embeddings=dir_vocab_size, embedding_dim=dir_emb_dim, padding_idx=0)
        self.cast_emb = nn.Embedding(num_embeddings=cast_vocab_size, embedding_dim=cast_emb_dim, padding_idx=0)
        self.writer_emb = nn.Embedding(num_embeddings=writer_vocab_size, embedding_dim=writer_emb_dim, padding_idx=0)

        # pooling: mean pooling over valid positions
        # We'll compute masked mean in forward

        # numeric and genre processing
        self.numeric_bn = nn.BatchNorm1d(numeric_dim) if numeric_dim > 0 else nn.Identity()
        self.genre_bn = nn.BatchNorm1d(genre_dim) if genre_dim > 0 else nn.Identity()

        # shared MLP
        shared_input_dim = numeric_dim + genre_dim + dir_emb_dim + cast_emb_dim + writer_emb_dim
        layers = []
        cur = shared_input_dim
        for h in shared_hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur = h
        self.shared = nn.Sequential(*layers)

        # heads
        # classification head
        self.cls_head = nn.Sequential(
            nn.Linear(cur, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # rating regression head
        self.rating_head = nn.Sequential(
            nn.Linear(cur, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # votes regression head (we predict log(numVotes+1))
        self.votes_head = nn.Sequential(
            nn.Linear(cur, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()   # MAE
        # optim params
        self.lr = lr
        self.weight_decay = weight_decay
        self.cls_weight = cls_weight
        self.rating_weight = rating_weight
        self.votes_weight = votes_weight

    def forward(self, numeric, genre, dirs, cast, writers):
        """
        Inputs:
          numeric: (B, numeric_dim)
          genre: (B, genre_dim)
          dirs: (B, Ld) LongTensor
          cast: (B, Lc) LongTensor
          writers: (B, Lw) LongTensor
        """
        B = numeric.size(0)

        # embeddings + masked mean pooling
        def emb_mean(emb_layer, ids):
            if ids.numel() == 0:
                # no tokens present in this batch for that field
                # return zeros
                return emb_layer(torch.zeros((B,1), dtype=torch.long, device=ids.device)).mean(dim=1)
            mask = (ids != 0).unsqueeze(-1).float()   # zero is padding/UNK
            emb = emb_layer(ids)                      # (B, L, D)
            summed = (emb * mask).sum(dim=1)          # (B, D)
            denom = mask.sum(dim=1).clamp(min=1.0)    # (B,1)
            mean = summed / denom
            return mean

        dir_vec = emb_mean(self.dir_emb, dirs)
        cast_vec = emb_mean(self.cast_emb, cast)
        writer_vec = emb_mean(self.writer_emb, writers) if writers.numel() != 0 else torch.zeros((B, self.hparams.writer_emb_dim), device=numeric.device)

        # numeric + genre BN
        numeric = self.numeric_bn(numeric)
        genre = self.genre_bn(genre)

        # concat
        x = torch.cat([numeric, genre, dir_vec, cast_vec, writer_vec], dim=1)
        shared = self.shared(x)

        # heads
        cls_logits = self.cls_head(shared).squeeze(1)     # (B,)
        rating_out = self.rating_head(shared).squeeze(1)  # (B,)
        votes_out = self.votes_head(shared).squeeze(1)    # (B,)

        return cls_logits, rating_out, votes_out

    def training_step(self, batch, batch_idx):
        numeric = batch['numeric'].float()
        genre = batch['genre'].float()
        dirs = batch['dirs'].long().to(numeric.device)
        cast = batch['cast'].long().to(numeric.device)
        writers = batch['writers'].long().to(numeric.device)

        cls_logits, rating_pred, votes_pred = self.forward(numeric, genre, dirs, cast, writers)

        # targets
        is_risky = batch['is_risky'].squeeze(1).to(numeric.device)
        avg_rating = batch['avg_rating'].squeeze(1).to(numeric.device)
        num_votes = batch['num_votes'].squeeze(1).to(numeric.device)
        log_votes = torch.log1p(num_votes)

        # losses
        loss_cls = self.bce_loss(cls_logits, is_risky)
        loss_rating = self.l1_loss(rating_pred, avg_rating)
        loss_votes = self.l1_loss(votes_pred, log_votes)

        loss = self.cls_weight * loss_cls + self.rating_weight * loss_rating + self.votes_weight * loss_votes

        # logging
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/loss_cls', loss_cls, on_epoch=True)
        self.log('train/loss_rating', loss_rating, on_epoch=True)
        self.log('train/loss_votes', loss_votes, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        numeric = batch['numeric'].float()
        genre = batch['genre'].float()
        dirs = batch['dirs'].long().to(numeric.device)
        cast = batch['cast'].long().to(numeric.device)
        writers = batch['writers'].long().to(numeric.device)

        cls_logits, rating_pred, votes_pred = self.forward(numeric, genre, dirs, cast, writers)

        is_risky = batch['is_risky'].squeeze(1).to(numeric.device)
        avg_rating = batch['avg_rating'].squeeze(1).to(numeric.device)
        num_votes = batch['num_votes'].squeeze(1).to(numeric.device)
        log_votes = torch.log1p(num_votes)

        loss_cls = self.bce_loss(cls_logits, is_risky)
        loss_rating = self.l1_loss(rating_pred, avg_rating)
        loss_votes = self.l1_loss(votes_pred, log_votes)
        loss = self.cls_weight * loss_cls + self.rating_weight * loss_rating + self.votes_weight * loss_votes

        # convert preds for metric logging (on CPU)
        preds_cls = torch.sigmoid(cls_logits).detach().cpu().numpy()
        preds_rating = rating_pred.detach().cpu().numpy()
        preds_votes = votes_pred.detach().cpu().numpy()
        targets_cls = is_risky.detach().cpu().numpy()
        targets_rating = avg_rating.detach().cpu().numpy()
        targets_votes = log_votes.detach().cpu().numpy()

        # compute simple metrics (batch-level)
        # log some aggregate metrics to TensorBoard/CSV
        # NOTE: for stable epoch metrics compute after epoch using validation_epoch_end
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        return {
            'loss': loss.detach(),
            'preds_cls': preds_cls,
            'targets_cls': targets_cls,
            'preds_rating': preds_rating,
            'targets_rating': targets_rating,
            'preds_votes': preds_votes,
            'targets_votes': targets_votes
        }

    def validation_epoch_end(self, outputs):
        # Collect all preds and targets
        preds_cls = np.concatenate([o['preds_cls'] for o in outputs])
        targets_cls = np.concatenate([o['targets_cls'] for o in outputs])
        preds_rating = np.concatenate([o['preds_rating'] for o in outputs])
        targets_rating = np.concatenate([o['targets_rating'] for o in outputs])
        preds_votes = np.concatenate([o['preds_votes'] for o in outputs])
        targets_votes = np.concatenate([o['targets_votes'] for o in outputs])

        # classification metrics
        try:
            auc = roc_auc_score(targets_cls, preds_cls)
        except Exception:
            auc = float('nan')
        preds_bin = (preds_cls >= 0.5).astype(int)
        f1 = f1_score(targets_cls, preds_bin, zero_division=0)

        # regression metrics (rating)
        mae_rating = mean_absolute_error(targets_rating, preds_rating)
        rmse_rating = math.sqrt(mean_squared_error(targets_rating, preds_rating))

        # votes (on log scale)
        mae_votes = mean_absolute_error(targets_votes, preds_votes)
        rmse_votes = math.sqrt(mean_squared_error(targets_votes, preds_votes))

        self.log('val/auc', auc, prog_bar=True)
        self.log('val/f1', f1)
        self.log('val/mae_rating', mae_rating)
        self.log('val/rmse_rating', rmse_rating)
        self.log('val/mae_votes', mae_votes)
        self.log('val/rmse_votes', rmse_votes)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optional: scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


# ---------------------------
# TRAINING FUNCTION
# ---------------------------
def train_hybrid_dl_models(
    data_path: str,
    save_dir: str = "models/hybrid_dl",
    topk_dir: int = 5000,
    topk_cast: int = 20000,
    topk_writer: int = 5000,
    batch_size: int = 256,
    max_epochs: int = 20,
    lr: float = 1e-3,
    gpu: Optional[int] = 0
):
    """
    End-to-end training for hybrid model using preprocessed DataFrame from preprocess_pipeline(data_path).
    - data_path: path to raw IMDb folder (preprocess_pipeline will be called) OR path to preprocessed CSV.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ----------------
    # 1. LOAD / PREPROCESS
    # ----------------
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = preprocess_pipeline(data_path)

    # ensure genre list exists and encoded
    df = encode_genres(df)
    # build list of genre columns
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    # numeric columns for the model (pick from available)
    numeric_cols = []
    for c in ['director_mean_rating', 'director_total_films', 'cast_mean_rating', 'cast_total_films', 'runtimeMinutes', 'startYear']:
        if c in df.columns:
            numeric_cols.append(c)
    # fill NA numeric
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # create log numVotes if missing
    if 'numVotes' not in df.columns:
        df['numVotes'] = 0
    df['log_numVotes'] = np.log1p(df['numVotes'].fillna(0))

    # drop rows with no person info? we can keep but model handles empty lists
    df = df.reset_index(drop=True)

    # ----------------
    # 2. Build vocabularies (top-k)
    # ----------------
    dir_lists = df['directors_nconst'].tolist() if 'directors_nconst' in df else [[] for _ in range(len(df))]
    cast_lists = df['cast_nconst'].tolist() if 'cast_nconst' in df else [[] for _ in range(len(df))]
    writer_lists = df['writers_nconst'].tolist() if 'writers_nconst' in df else [[] for _ in range(len(df))]

    dir_vocab = build_topk_vocab(dir_lists, topk_dir, unk_token="<UNK>")
    cast_vocab = build_topk_vocab(cast_lists, topk_cast, unk_token="<UNK>")
    writer_vocab = build_topk_vocab(writer_lists, topk_writer, unk_token="<UNK>")

    # Save vocabs for inference
    with open(os.path.join(save_dir, "dir_vocab.json"), "w") as f:
        json.dump(dir_vocab, f)
    with open(os.path.join(save_dir, "cast_vocab.json"), "w") as f:
        json.dump(cast_vocab, f)
    with open(os.path.join(save_dir, "writer_vocab.json"), "w") as f:
        json.dump(writer_vocab, f)

    # ----------------
    # 3. Dataset split
    # ----------------
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RND_SEED, stratify=df['is_risky'] if 'is_risky' in df else None)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=RND_SEED, stratify=test_df['is_risky'] if 'is_risky' in test_df else None)

    # ----------------
    # 4. Create Datasets & DataLoaders
    # ----------------
    dataset_train = IMDBHybridDataset(train_df, dir_vocab, cast_vocab, writer_vocab, numeric_cols, genre_cols)
    dataset_val = IMDBHybridDataset(val_df, dir_vocab, cast_vocab, writer_vocab, numeric_cols, genre_cols)
    dataset_test = IMDBHybridDataset(test_df, dir_vocab, cast_vocab, writer_vocab, numeric_cols, genre_cols)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=hybrid_collate_fn, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=hybrid_collate_fn, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=hybrid_collate_fn, pin_memory=True)

    # ----------------
    # 5. Instantiate model
    # ----------------
    numeric_dim = len(numeric_cols)
    genre_dim = len(genre_cols)
    model = HybridMovieModel(
        numeric_dim=numeric_dim,
        genre_dim=genre_dim,
        dir_vocab_size=len(dir_vocab),
        cast_vocab_size=len(cast_vocab),
        writer_vocab_size=len(writer_vocab),
        dir_emb_dim=32,
        cast_emb_dim=64,
        writer_emb_dim=32,
        shared_hidden_dims=[256, 128],
        dropout=0.2,
        lr=lr,
        weight_decay=1e-5,
        cls_weight=1.0,
        rating_weight=1.0,
        votes_weight=1.0
    )

    # ----------------
    # 6. Trainer & Callbacks
    # ----------------
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=save_dir,
        filename="hybrid-{epoch:02d}-{val/loss:.4f}",
        save_top_k=1,
        mode="min"
    )
    early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min", verbose=True)
    logger = CSVLogger(save_dir, name="hybrid_logs")

    trainer_kwargs = dict(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        precision=16,          # mixed precision
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )
    if torch.cuda.is_available() and gpu is not None:
        trainer_kwargs.update({"accelerator": "gpu", "devices": [gpu]})
    else:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

    trainer = pl.Trainer(**trainer_kwargs)

    # ----------------
    # 7. Fit
    # ----------------
    trainer.fit(model, train_loader, val_loader)

    # ----------------
    # 8. Load best checkpoint and evaluate on test set
    # ----------------
    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path == "":
        print("No checkpoint found, using current model weights.")
        best_model = model
    else:
        best_model = HybridMovieModel.load_from_checkpoint(best_ckpt_path)

    # Put model in eval mode
    best_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() and gpu is not None else "cpu")
    best_model.to(device)

    # run inference on test dataset and compute metrics
    all_preds_cls = []
    all_targets_cls = []
    all_preds_rating = []
    all_targets_rating = []
    all_preds_votes = []
    all_targets_votes = []

    with torch.no_grad():
        for batch in test_loader:
            numeric = batch['numeric'].float().to(device)
            genre = batch['genre'].float().to(device)
            dirs = batch['dirs'].long().to(device)
            cast = batch['cast'].long().to(device)
            writers = batch['writers'].long().to(device)

            cls_logits, rating_pred, votes_pred = best_model.forward(numeric, genre, dirs, cast, writers)
            probs = torch.sigmoid(cls_logits).cpu().numpy()
            rating_pred = rating_pred.cpu().numpy()
            votes_pred = votes_pred.cpu().numpy()
            is_risky = batch['is_risky'].squeeze(1).cpu().numpy()
            avg_rating = batch['avg_rating'].squeeze(1).cpu().numpy()
            log_votes = np.log1p(batch['num_votes'].squeeze(1).cpu().numpy())

            all_preds_cls.append(probs)
            all_targets_cls.append(is_risky)
            all_preds_rating.append(rating_pred)
            all_targets_rating.append(avg_rating)
            all_preds_votes.append(votes_pred)
            all_targets_votes.append(log_votes)

    preds_cls = np.concatenate(all_preds_cls)
    targets_cls = np.concatenate(all_targets_cls)
    preds_rating = np.concatenate(all_preds_rating)
    targets_rating = np.concatenate(all_targets_rating)
    preds_votes = np.concatenate(all_preds_votes)
    targets_votes = np.concatenate(all_targets_votes)

    # classification metrics
    try:
        auc = roc_auc_score(targets_cls, preds_cls)
    except Exception:
        auc = float('nan')
    preds_bin = (preds_cls >= 0.5).astype(int)
    f1 = f1_score(targets_cls, preds_bin, zero_division=0)

    # regression metrics
    mae_rating = mean_absolute_error(targets_rating, preds_rating)
    rmse_rating = math.sqrt(mean_squared_error(targets_rating, preds_rating))
    mae_votes = mean_absolute_error(targets_votes, preds_votes)
    rmse_votes = math.sqrt(mean_squared_error(targets_votes, preds_votes))

    metrics = {
        'test_auc': float(auc),
        'test_f1': float(f1),
        'test_mae_rating': float(mae_rating),
        'test_rmse_rating': float(rmse_rating),
        'test_mae_votes_log': float(mae_votes),
        'test_rmse_votes_log': float(rmse_votes)
    }

    print("TEST METRICS:", json.dumps(metrics, indent=2))

    # ----------------
    # 9. Save model (checkpoint already saved). Also export TorchScript for inference
    # ----------------
    final_ckpt = os.path.join(save_dir, "hybrid_final.ckpt")
    trainer.save_checkpoint(final_ckpt)

    # TorchScript export (example - requires a small example input)
    # Prepare a dummy batch for tracing (use small batch)
    sample_batch = next(iter(test_loader))
    sample_numeric = sample_batch['numeric'].float().to(device)
    sample_genre = sample_batch['genre'].float().to(device)
    sample_dirs = sample_batch['dirs'].long().to(device)
    sample_cast = sample_batch['cast'].long().to(device)
    sample_writers = sample_batch['writers'].long().to(device)

    model_for_export = best_model.to(device)
    model_for_export.eval()
    try:
        scripted = torch.jit.trace(model_for_export, (sample_numeric, sample_genre, sample_dirs, sample_cast, sample_writers))
        scripted.save(os.path.join(save_dir, "hybrid_model.ts"))
        print("TorchScript model saved.")
    except Exception as e:
        print("TorchScript export failed:", e)

    # Save final metrics
    with open(os.path.join(save_dir, "hybrid_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        'best_ckpt': best_ckpt_path,
        'metrics': metrics,
        'vocab_paths': {
            'dir_vocab': os.path.join(save_dir, "dir_vocab.json"),
            'cast_vocab': os.path.join(save_dir, "cast_vocab.json"),
            'writer_vocab': os.path.join(save_dir, "writer_vocab.json")
        }
    }


