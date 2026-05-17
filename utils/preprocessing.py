# utils/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── These are the exact columns the models expect ──────────────────
FEATURE_COLUMNS = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'Geography_Germany',
    'Geography_Spain', 'Gender_Male'
]

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV and drop irrelevant columns.
    """
    df = pd.read_csv(filepath)
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    return df


def fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip Age outliers using IQR method — same as midterm.
    """
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['Age'] = np.clip(df['Age'], lower, upper)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode Geography and Gender — same as midterm.
    drop_first=True to avoid dummy variable trap.
    """
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split into X (features) and y (target).
    """
    X = df[FEATURE_COLUMNS]
    y = df['Exited']
    return X, y


def preprocess_single_input(raw_input: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Preprocess a single user input dictionary for inference.
    raw_input example:
    {
        'CreditScore': 650,
        'Age': 35,
        'Tenure': 5,
        'Balance': 75000,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 90000,
        'Geography': 'France',
        'Gender': 'Male'
    }
    """
    # Start with zeros for all feature columns
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Fill in direct numerical features
    direct_cols = [
        'CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    for col in direct_cols:
        if col in raw_input:
            row[col] = raw_input[col]

    # Handle Geography encoding
    geo = raw_input.get('Geography', 'France')
    if geo == 'Germany':
        row['Geography_Germany'] = 1
    elif geo == 'Spain':
        row['Geography_Spain'] = 1
    # France → both stay 0 (drop_first baseline)

    # Handle Gender encoding
    gender = raw_input.get('Gender', 'Female')
    if gender == 'Male':
        row['Gender_Male'] = 1

    # Convert to numpy array and scale
    X = np.array([list(row.values())])
    X_scaled = scaler.transform(X)
    return X_scaled


def full_preprocessing_pipeline(filepath: str):
    """
    Complete pipeline used in train_classifiers.py.
    Returns: X, y, scaler
    """
    df = load_and_clean(filepath)
    df = fix_outliers(df)
    df = encode_categoricals(df)
    X, y = get_features_and_target(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler