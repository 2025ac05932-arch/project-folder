import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from model.logistic_regression import create_model as create_logistic
from model.decision_tree import create_model as create_tree
from model.knn import create_model as create_knn
from model.naive_bayes import create_model as create_nb
from model.random_forest import create_model as create_rf


def load_dataset():
    files = glob.glob("data/*.csv")
    if not files:
        raise FileNotFoundError("No CSV found in data/")
    return pd.read_csv(files[0])


def prepare_data(df):
    df = df.copy()
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    drop_cols = ["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    return df.drop(columns=drop_cols, errors="ignore"), df["Attrition"].astype(int)


def make_preprocessor(X):
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ])


def get_models():
    return {
        "Logistic Regression": create_logistic(),
        "Decision Tree": create_tree(),
        "KNN": create_knn(),
        "Naive Bayes": create_nb(),
        "Random Forest": create_rf(),
    }


def train_models(save_test_data=False):
    df = load_dataset()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    results, trained = [], {}
    for name, estimator in get_models().items():
        pipe = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        trained[name] = pipe
    results_df = pd.DataFrame(results)
    if save_test_data:
        test_data = X_test.copy()
        test_data["Attrition"] = y_test.map({1: "Yes", 0: "No"})
        test_data.to_csv("test_data.csv", index=False)
        results_df.to_csv("model/model_comparison.csv", index=False)
    return results_df, trained


if __name__ == "__main__":
    results, _ = train_models(save_test_data=True)
    print(results.round(4).to_string(index=False))
