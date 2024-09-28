import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
import joblib

# load cleaned data
df = pd.read_pickle('./ML models/cleaned_data2.pkl')

# initialize text vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# extract features and targets 
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

# split data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

# define model
model = RandomForestClassifier(class_weight='balanced', random_state=3)

# define hyperparameter grid 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# hyperparameter tuning using grid search on training set
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=30, scoring='f1_weighted', cv=5, verbose=1)
random_search.fit(X_train, y_train)

# examine grid search results 
print(f'Best Hyperparameters: {random_search.best_params_}')
print(f'Best Cross-Validation F1 Score: {random_search.best_score_}')

# extract best model
best_model = random_search.best_estimator_

# save vectorizer and trained model
joblib.dump(vectorizer, './ML models/Random Forest/tfidf_vectorizer.pkl')
joblib.dump(best_model, './ML models/Random Forest/rand_for_model.pkl')

# evaluate model on test set 
y_pred = best_model.predict(X_test)

# examine accuracy score, classification report, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")

report = classification_report(y_test, y_pred)
print(f"Classification Report:\n {report}")

cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
