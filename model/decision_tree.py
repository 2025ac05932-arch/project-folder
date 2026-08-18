from sklearn.tree import DecisionTreeClassifier

def create_model():
    return DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
