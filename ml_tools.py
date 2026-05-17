# ml_tools.py
import joblib
import numpy as np
from langchain_core.tools import tool
from utils.preprocessing import preprocess_single_input

# ── 1. Load All Saved Models ──────────────────────────────────────────
print("📦 Loading models...")
svm_bundle = joblib.load("models/svm_model.pkl")
dt_bundle  = joblib.load("models/dt_model.pkl")
nn_bundle  = joblib.load("models/nn_model.pkl")
print("✅ All models loaded!")


# ── 2. Helper Function ────────────────────────────────────────────────
def _infer(bundle: dict, raw_input: dict) -> str:
    try:
        scaler = bundle["scaler"]
        model  = bundle["model"]

        X_scaled = preprocess_single_input(raw_input, scaler)
        pred     = model.predict(X_scaled)[0]
        proba    = model.predict_proba(X_scaled).max()
        label    = "CHURN ⚠️" if pred == 1 else "STAY ✅"

        return (
            f"Prediction: {label} | "
            f"Confidence: {proba:.2%} | "
            f"Raw class: {int(pred)}"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ── 3. Tool Definitions ───────────────────────────────────────────────
@tool
def classify_with_svm(
    CreditScore: int,
    Age: int,
    Tenure: int,
    Balance: float,
    NumOfProducts: int,
    HasCrCard: int,
    IsActiveMember: int,
    EstimatedSalary: float,
    Geography: str,
    Gender: str
) -> str:
    """
    Predict customer churn using Support Vector Machine (SVM).
    Use this for structured numerical customer data with clear class boundaries.
    Geography must be one of: France, Germany, Spain.
    Gender must be one of: Male, Female.
    HasCrCard and IsActiveMember must be 1 or 0.
    Returns churn prediction with confidence score.
    """
    raw_input = {
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        "Geography": Geography,
        "Gender": Gender
    }
    return _infer(svm_bundle, raw_input)


@tool
def classify_with_decision_tree(
    CreditScore: int,
    Age: int,
    Tenure: int,
    Balance: float,
    NumOfProducts: int,
    HasCrCard: int,
    IsActiveMember: int,
    EstimatedSalary: float,
    Geography: str,
    Gender: str
) -> str:
    """
    Predict customer churn using Decision Tree classifier.
    Use this when you need interpretable rule-based predictions.
    Best when NumOfProducts or IsActiveMember are key factors.
    Geography must be one of: France, Germany, Spain.
    Gender must be one of: Male, Female.
    HasCrCard and IsActiveMember must be 1 or 0.
    Returns churn prediction with confidence score.
    """
    raw_input = {
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        "Geography": Geography,
        "Gender": Gender
    }
    return _infer(dt_bundle, raw_input)


@tool
def classify_with_neural_network(
    CreditScore: int,
    Age: int,
    Tenure: int,
    Balance: float,
    NumOfProducts: int,
    HasCrCard: int,
    IsActiveMember: int,
    EstimatedSalary: float,
    Geography: str,
    Gender: str
) -> str:
    """
    Predict customer churn using Neural Network (MLP Classifier).
    Use this for complex non-linear patterns in customer behavior.
    This is the highest accuracy model — use it when precision matters most.
    Geography must be one of: France, Germany, Spain.
    Gender must be one of: Male, Female.
    HasCrCard and IsActiveMember must be 1 or 0.
    Returns churn prediction with confidence score.
    """
    raw_input = {
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        "Geography": Geography,
        "Gender": Gender
    }
    return _infer(nn_bundle, raw_input)


# ── 4. Export ─────────────────────────────────────────────────────────
ALL_TOOLS = [
    classify_with_svm,
    classify_with_decision_tree,
    classify_with_neural_network,
]

print("🔗 All tools ready!")