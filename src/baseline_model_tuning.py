import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import BaseEstimator, RegressorMixin
import logging
import os

from task import MODEL_FEATURE_COLUMNS, LABEL_COLUMN, DATA_PATH, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GAN_D_Wrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden1_size=32, hidden2_size=16, learning_rate=0.001, epochs=10, batch_size=32, threshold_percentile=50.0):
        self.input_dim = input_dim
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.discriminator = self._build_discriminator()
        self.threshold_ = 0.0

    def _build_discriminator(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden1_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden1_size, self.hidden2_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden2_size, 1),
            nn.Sigmoid()
        )

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        self.discriminator.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.discriminator(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.discriminator.eval()
        with torch.no_grad():
            outputs = self.discriminator(X_tensor)
            anomaly_scores = 1.0 - outputs.numpy().flatten()
            self.threshold_ = np.percentile(anomaly_scores, self.threshold_percentile)
        
        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.discriminator.eval()
        with torch.no_grad():
            outputs = self.discriminator(X_tensor)
            anomaly_scores = 1.0 - outputs.numpy().flatten()
        return (anomaly_scores > self.threshold_).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return f1_score(y, y_pred)

def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average='binary')

def main():
    logger.info("Starting baseline model hyperparameter tuning...")

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Dataset not found at {DATA_PATH}. Please check the path.")
        return

    df = df.sample(frac=0.1, random_state=RANDOM_SEED)
    logger.info(f"Sampled 10% of the data. New shape: {df.shape}")

    for col in MODEL_FEATURE_COLUMNS + [LABEL_COLUMN]:
        if col in df.columns:
            df[col] = df[col].replace('-', 0)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"Column '{col}' not found in the DataFrame.")

    df[MODEL_FEATURE_COLUMNS] = df[MODEL_FEATURE_COLUMNS].fillna(0)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna(0)

    X = df[MODEL_FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Data preprocessed. Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    model = GAN_D_Wrapper(input_dim=X_train.shape[1])
    
    param_grid = {
        'hidden1_size': [32, 64, 128],
        'hidden2_size': [16, 32, 64],
        'learning_rate': [0.001, 0.0001],
        'epochs': [10, 20],
        'batch_size': [32, 64],
        'threshold_percentile': [40, 50, 60]
    }
    
    scorer = make_scorer(custom_scorer, greater_is_better=True)

    logger.info("Starting GridSearchCV...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=3, n_jobs=-1, verbose=2)
    
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"An error occurred during GridSearchCV fitting: {e}", exc_info=True)
        return

    logger.info("GridSearchCV finished.")

    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best F1-score on validation set: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    print(f"\nF1-score of the best model on the test set: {test_f1:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)
    
    output_dir = "tuning_results"
    os.makedirs(output_dir, exist_ok=True)
    results_filename = os.path.join(output_dir, "gan_discriminator_grid_search_results.csv")
    results_df.to_csv(results_filename, index=False)
    logger.info(f"Grid search results saved to {results_filename}")

    best_params_filename = os.path.join(output_dir, "best_hyperparameters.txt")
    with open(best_params_filename, 'w') as f:
        f.write("Best parameters found for GAN Discriminator:\n")
        f.write(str(grid_search.best_params_))
        f.write(f"\n\nBest F1-score on validation set: {grid_search.best_score_:.4f}\n")
        f.write(f"F1-score on test set: {test_f1:.4f}\n")
    logger.info(f"Best parameters saved to {best_params_filename}")

if __name__ == "__main__":
    main() 