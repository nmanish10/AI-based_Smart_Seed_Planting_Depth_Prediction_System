import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import joblib

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
file_path = r"C:\Users\nmani\Downloads\synthetic_seed_depth_dataset.csv"
df = pd.read_csv(file_path)

# Encode categorical features (if any categorical features need to be encoded)
le = LabelEncoder()
soil_le = LabelEncoder()
df['Soil Type'] = soil_le.fit_transform(df['Soil Type'])
df['Seed Type'] = le.fit_transform(df['Seed Type'])

# Define features and target variable
x = df.drop('Depth (inches)', axis=1)  # Features
y = df['Depth (inches)']  # Target column

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Regression models dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor (SVR)": SVR(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
    "LightGBM Regressor": LGBMRegressor(random_state=42),
    "XGBoost Regressor": XGBRegressor(random_state=42),
    "CatBoost Regressor": CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, random_state=42, silent=True),
    "MLP Regressor": MLPRegressor(random_state=42)
}

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

param_grid_lr = {
    'fit_intercept': [True, False]
}

# Hyperparameter tuning for Random Forest
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, n_jobs=-1, verbose=1)
best_rf = grid_rf.fit(x_train, y_train).best_estimator_

# Hyperparameter tuning for Linear Regression
grid_lr = GridSearchCV(LinearRegression(), param_grid_lr, cv=5, n_jobs=-1, verbose=1)
best_lr = grid_lr.fit(x_train, y_train).best_estimator_

# Train and evaluate all models using cross-validation
model_performance = {}
for name, model in models.items():
    print(f"\nTraining {name}...")

    # Cross-validation for robust evaluation
    cross_val_accuracy = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    model_performance[name] = cross_val_accuracy
    
    print(f"{name} Cross-Validation MSE: {cross_val_accuracy:.4f}")

# Sort models by performance and select the top 4
top_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)[:4]
best_models = [(name, models[name]) for name, _ in top_models]

# Ensemble Voting Regressor with best models
voting_regressor = VotingRegressor(estimators=best_models)

voting_regressor.fit(x_train, y_train)
y_pred_voting = voting_regressor.predict(x_test)
voting_mse = mean_squared_error(y_test, y_pred_voting)
voting_r2 = r2_score(y_test, y_pred_voting)

print(f"\nEnsemble Voting Regressor MSE: {voting_mse:.4f}")
print(f"Ensemble Voting Regressor R²: {voting_r2:.4f}")

# Save the best models and the Voting Regressor
joblib.dump(voting_regressor, 'voting_regressor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(soil_le, 'soil_label_encoder.pkl')

# Optionally, save individual best models if needed
for name, model in best_models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    model_mse = mean_squared_error(y_test, y_pred)
    model_r2 = r2_score(y_test, y_pred)
    joblib.dump(model, f'{name}_regressor_model.pkl')
    print(f"{name} - MSE: {model_mse:.4f}, R²: {model_r2:.4f}")

# Load a saved model (example)
# loaded_model = joblib.load('voting_regressor_model.pkl')
