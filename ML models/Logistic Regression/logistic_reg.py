import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
import joblib

# load cleaned data
df = pd.read_pickle('./ML models/cleaned_data.pkl')

# input text vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# sentiment encoding 
label_encoder = LabelEncoder()  

# extract features and targets
X = vectorizer.fit_transform(df['clean_text']) # sparse matrix of TF-IDF values 
y = label_encoder.fit_transform(df['sentiment']) # negative - 0 neutral - 1 positive - 2 

# define model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# define hyperparameter grid 
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'class_weight': [None, 'balanced']
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
joblib.dump(vectorizer, './ML models/Logistic Regression/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, './ML models/Logistic Regression/label_encoder.pkl')
joblib.dump(grid_search.best_estimator_, './ML models/Logistic Regression/log_reg_model.pkl')

# extract best model
best_model = grid_search.best_estimator_

# generate cross-validation predictions 
y_pred = cross_val_predict(best_model, X, y, kf)

# examine accuracy score, classification report, and confusion matrix
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy Score:", accuracy)
report = classification_report(y, y_pred, target_names=label_encoder.classes_)
print(f"Classification Report:\n", report)
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



