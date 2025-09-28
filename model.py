# model.py
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
import joblib

def get_models(random_state=42):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=random_state),
        "SGD_Logistic": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_state)
    }
    return models

def train_and_evaluate(models, X_train, y_train, X_test, y_test, label_encoder):
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4)
        
        results[name] = {
            "model": mdl,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "report": report,
            "y_pred": y_pred
        }
    best_name = max(results.keys(), key=lambda k: results[k]["f1_macro"])
    best_model = results[best_name]["model"]
    return best_name, best_model, results

def save_pipeline(preprocessor, label_encoder, model, filename="planetary_model_pipeline_final.joblib"):
    save_dict = {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "model": model
    }
    joblib.dump(save_dict, filename)
