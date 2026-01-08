import pandas as pd
import joblib

problems = pd.read_json('/workspaces/autojudge_acm/problems_data.jsonl', lines=True)
problems = problems[["title","description","input_description","output_description","problem_class","problem_score"]]


import joblib
import pandas as pd
import re
import string

def preprocess_autojudge_data(problems):
    # Create a copy to avoid SettingWithCopyWarning
    problems = problems.copy()
    
    #  Combine specific text fields
    problems['combined_text'] = (
        problems['title'].fillna('') + " " + 
        problems['description'].fillna('') + " " + 
        problems['input_description'].fillna('') + " " + 
        problems['output_description'].fillna('')
    )
    
    #  Basic Cleaning
    def clean_logic(text):
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    problems['cleaned_text'] = problems['combined_text'].apply(clean_logic)
    
    #  Drop any rows where targets are missing
    problems = problems.dropna(subset=['problem_class', 'problem_score'])
    
    return problems

problems = preprocess_autojudge_data(problems)






import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

#  Feature Extraction 
print("Extracting features from 'problems' table...")
# ngram_range=(1, 2) helps the model recognize phrases like "dynamic programming"
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
X = tfidf.fit_transform(problems['cleaned_text'])

#  Target Variables 
# Classification Target (Easy/Medium/Hard)
y_class = problems['problem_class']

# Regression Target (Numerical Score)
y_score = problems['problem_score']

#  Train/Test Split 
# split both targets at once using the same random_state for consistency
X_train, X_test, y_c_train, y_c_test, y_s_train, y_s_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42
)

#  Model 1 - Classification 
print("Training Classification Model (Random Forest)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_c_train)

#  Model 2 - Regression 
print("Training Regression Model (Random Forest)...")
reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
reg.fit(X_train, y_s_train)

#  Evaluation 
print("\n" + "="*30)
print("CLASSIFICATION REPORT")
print("="*30)
c_preds = clf.predict(X_test)
print(classification_report(y_c_test, c_preds))

print("\n" + "="*30)
print("REGRESSION METRICS")
print("="*30)
s_preds = reg.predict(X_test)
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_s_test, s_preds):.2f}")
print(f"R2 Score (Variance Explained): {r2_score(y_s_test, s_preds):.2f}")

#  Save Models for Web UI
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(clf, 'clf_model.pkl')
joblib.dump(reg, 'reg_model.pkl')

print("\nAll models and vectorizers saved successfully!")


import pickle

pickle.dump(tfidf, open('some.pkl', 'wb'))