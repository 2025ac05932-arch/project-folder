from sklearn.ensemble import RandomForestClassifier

def create_model():
    return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
