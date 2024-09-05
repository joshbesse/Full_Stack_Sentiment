import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
import joblib

# load cleaned data
df = pd.read_pickle('./ML models/cleaned_data.pkl')

# initialize text vectorizer
vectorizer = TfidfVectorizer()

# initialize label encoder 
label_encoder = LabelEncoder()

# extract features and targets 
X = vectorizer.fit_transform(df['clean_text'])
y = label_encoder.fit_transform(df['sentiment'])

# define model
model = RandomForestClassifier(class_weight='balanced', random_state=3)

# define hyperparameter grid 
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# initialize stratified k-fold cross-validation and grid search
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', n_jobs=1, cv=kf, verbose=2)

# perform grid search 
grid_search.fit(X, y)

# examine results 
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Best Cross-Validation F1 Score: {grid_search.best_score_}')

# save vectorizer, label encoder, and trained model
joblib.dump(vectorizer, './ML models/Random Forest/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, './ML models/Random Forest/label_encoder.pkl')
joblib.dump(grid_search.best_estimator_, './ML models/Random Forest/rand_for_model.pkl')

# extract best model
best_model = grid_search.best_estimator_

# generate cross-validation predictions
y_pred = cross_val_predict(estimator=best_model, X=X, y=y, cv=kf)

# examine accuracy score, classification report, and confusion matrix
accuracy = accuracy_score(y, y_pred)
print("Accuracy Score:", accuracy)
report = classification_report(y, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
