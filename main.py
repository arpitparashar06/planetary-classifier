# main.py
from sklearn.model_selection import train_test_split
from preprocessing import load_data, build_preprocessor, encode_labels
from model import get_models, train_and_evaluate, save_pipeline

CSV_PATH = "data/cosmicclassifierTraining.txt.csv"
LABEL_COL = "Prediction"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Load and preprocess data
X_raw, y_raw = load_data(CSV_PATH, LABEL_COL)
preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_raw)
y, label_encoder = encode_labels(y_raw)

# Split dataset
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Fit preprocessor
preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Train models
models = get_models(RANDOM_STATE)
best_name, best_model, results = train_and_evaluate(models, X_train, y_train, X_test, y_test, label_encoder)

# Save best model pipeline
save_pipeline(preprocessor, label_encoder, best_model)

# Print summary
print(f"Best model chosen: {best_name}")
for name, res in results.items():
    print(f"{name} -> Accuracy: {res['accuracy']:.4f}, Macro F1: {res['f1_macro']:.4f}")
