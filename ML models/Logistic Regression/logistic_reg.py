import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
import joblib

# load cleaned data
df = pd.read_pickle('./ML models/cleaned_data2.pkl')

# input text vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# extract features and targets
X = vectorizer.fit_transform(df['clean_text']) # sparse matrix of TF-IDF values 
y = df['category'] # -1: negative, 0: neutral, 1: positive

# split data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

# define model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# define hyperparameter grid 
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'class_weight': [None, 'balanced']
}

# hyperparameter tuning using grid search on training set  
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', n_jobs=1, cv=5, verbose=2)
grid_search.fit(X_train, y_train)

# examine grid search results
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Best Cross-Validation F1 Score: {grid_search.best_score_}')

# extract best model
best_model = grid_search.best_estimator_

# save vectorizer and trained model
joblib.dump(vectorizer, './ML models/Logistic Regression/tfidf_vectorizer.pkl')
joblib.dump(grid_search.best_estimator_, './ML models/Logistic Regression/log_reg_model.pkl')

# evaluate model on test set 
y_pred = best_model.predict(X_test)

# examine accuracy score, classification report, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
report = classification_report(y_test, y_pred)
print(f"Classification Report:\n {report}")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



