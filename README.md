# Credit Card Fraud Detection — ML Models Playground

A local Streamlit app for experimenting with a Decision Tree classifier.

## 1. Prepare the dataset
Use a CSV named `creditcard.csv` with a target column named `Class`:
- `0` = legitimate transaction
- `1` = fraudulent transaction

You can put the CSV in this project folder, or upload it in the app sidebar.

## 2. Create and activate a virtual environment (recommended)

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install packages
```bash
pip install -r requirements.txt
```

## 4. Run the app
```bash
streamlit run app.py
```

Streamlit prints a local URL (usually http://localhost:8501). Open it in your browser.

## Features
- Criterion: gini, entropy, log_loss
- Max depth: 1–100 or unlimited (`None`)
- Min samples split: 2–500 (scikit-learn requires at least 2)
- Min samples leaf: 1–500
- Training strategy: no balancing, class weights, or SMOTE
- Fraud probability threshold
- Accuracy, precision, recall, F1, average precision
- Confusion matrix and classification report
- Decision tree preview up to 5 levels, with deeper branches omitted
- Downloadable classification report

## Important evaluation notes
- The train/test split is stratified and fixed with `random_state=42` for fair hyperparameter comparisons.
- SMOTE and class weighting are applied only to training data; the test set remains untouched.
- The app is an educational experiment, not a production banking fraud system.
