import pandas as pd 
import matplotlib.pyplot as plt 

import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def explore_data(df):
    print(df.head())
    print(df.columns) # Year, Month, Day, Time of Tweet (morning, noon, night), text, sentiment (negative, neutral, positive), Platform
    print(df.shape) # rows: 499, columns: 7
    print(df.info()) # no missing values 
    print(df.isnull().sum()) # no null values
    print(df.duplicated().value_counts()) # there are 105 duplicate rows 
    print(df[df.duplicated()]['sentiment'].value_counts()) # of duplicate rows -> negative: 25 (23.80%) neutral: 46 (43.80%) positive: 34 (32.38%)
    print(df['sentiment'].value_counts()) # negative: 134 (26.85%) neutral: 199 (39.87%) positive: 166 (33.26%)

def prepare_text(text):
    # convert text to lowercase
    text = text.lower()

    # remove punctuation, numbers, and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # tokenize the text
    tokens = word_tokenize(text)

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # join tokens back into a single string 
    cleaned_text = ' '.join(tokens)
    return tokens, cleaned_text

# download punkt tokenizer data
nltk.download('punkt')
nltk.download('wordnet')

# load csv file 
df = pd.read_csv('./ML model/sentiment_analysis.csv')

# remove duplicate rows 
df = df.drop_duplicates()

# clean the input text 
df[['tokens', 'clean_text']] = df['text'].apply(prepare_text).apply(pd.Series)

# save cleaned data as pickle file 
df.to_pickle('./ML model/cleaned_data.pkl')

# data visualization 
# sentiment class distribution
plt.bar(df['sentiment'].value_counts().index, df['sentiment'].value_counts().values, color=['darkgray', 'lawngreen', 'red'])
plt.xlabel('Sentiment Classes')
plt.ylabel('Class Counts')
plt.title('Sentiment Class Distribution')
plt.show()

# text length distribution
length_tokens = df['tokens'].apply(len)
plt.hist(length_tokens, bins=15)
plt.xlabel('Length of Text (in # of words)')
plt.ylabel('Text Counts')
plt.title('Text Length Distribution')
plt.show()
