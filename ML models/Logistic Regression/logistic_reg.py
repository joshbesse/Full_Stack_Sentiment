import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
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


