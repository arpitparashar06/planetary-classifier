# planetary-classifier
This project classifies planetary/cosmic data using machine learning.
The dataset includes both numeric and categorical features, and multiple preprocessing techniques were tested to select the best-performing one.
Three models were trained and compared: Logistic Regression, Random Forest, and SGD Classifier.
The final model pipeline, including preprocessing and label encoding, is saved for deployment.

Project Structure:
planetary-classifier/
│
├── data/
│   └── cosmicclassifierTraining.txt.csv
├── preprocessing.py       # Functions for preprocessing and feature transformations
├── model.py               # Code for training and evaluating classifiers
├── main.py                # Integrates preprocessing + training
├── planetary_model_pipeline_final.joblib   # Saved trained model pipeline
├── requirements.txt       # Project dependencies
└── README.md

HOW TO RUN:
1. Clone the repository:

git clone https://github.com/<your-username>/planetary-classifier.git
cd planetary-classifier


2. Install dependencies:

pip install -r requirements.txt


3. Run the project:

python main.py


4. The best model will be automatically saved as:

planetary_model_pipeline_final.joblib




RESULTS:
Compared 3 preprocessing techniques to handle numeric & categorical data

Trained with Logistic Regression, Random Forest, and SGD Classifier

Selected the best model based on macro F1-score

Best Model Accuracy: 91%

Confusion matrix and classification metrics are generated for detailed evaluation

Best Model: Random Forest
Accuracy: 91%
Macro F1-score: 0.88
Confusion Matrix:
[[30  2  1]
 [ 3 28  0]
 [ 0  2 29]]



TECH STACK:

Language: Python

Libraries: Pandas, NumPy, scikit-learn, joblib

Techniques: Data preprocessing pipelines, ML model comparison & selection, saving ML pipelines




