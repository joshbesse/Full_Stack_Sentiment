import pandas as pd 
import matplotlib.pyplot as plt 

# load datasets
redd_df = pd.read_csv('./ML models/Reddit_Data.csv')
twit_df = pd.read_csv('./ML models/Twitter_Data.csv')

# explore reddit df 
#print(redd_df.columns) # clean_comment, category (-1: negative, 0: neutral, 1: positive)
#print(redd_df.shape) # rows: 37249 columns: 2
#print(redd_df.info()) # clean_comment has null values, clean_comment: object, category: int64
#print(redd_df.isna().sum()) # clean_comment has 100 na values 
#print(redd_df.duplicated().value_counts()) # there are 449 duplicate rows 
#print(redd_df[redd_df.duplicated()]['category'].value_counts()) # 363 neutral duplicate rows, 59 positive duplicate rows, 27 negative duplicate rows
#print(redd_df['category'].value_counts()) # class imbalance -> 15830 positive, 13142 neutral, 8277 negative

# explore twitter df 
#print(twit_df.columns) # clean_text, category
#print(twit_df.shape) # rows: 162980 columns: 2
#print(twit_df.info()) # clean_text and category have null values, clean_text: object, category: float64
#print(twit_df.isna().sum()) # clean_text has 4 na values, category has 7 na values
#print(twit_df.duplicated().value_counts()) # there is 1 duplicate row 
#print(twit_df['category'].value_counts()) # class imbalance -> 72250 positive, 55213 neutral, 35510 negative

# remove rows with na values
redd_df = redd_df.dropna()
twit_df = twit_df.dropna()

# remove duplicate rows 
redd_df = redd_df.drop_duplicates()
twit_df = twit_df.drop_duplicates()

# rename reddit df columns to match twitter df 
redd_df.columns = ['clean_text', 'category']

# convert twitter df category to int 
twit_df['category'] = twit_df['category'].astype(int)

# join datasets 
df = pd.concat([redd_df, twit_df], ignore_index=True)

# save cleaned data as pickle file
df.to_pickle('./ML models/cleaned_data2.pkl')

# data visualization
# sentiment class distribution
plt.bar(df['category'].value_counts().index, df['category'].value_counts().values, color=['lawngreen', 'darkgray', 'red'])
plt.xlabel('Sentiment Classes')
plt.ylabel('Class Counts')
plt.title('Sentiment Class Distribution')
plt.show()

# text length distribution (in # of characters)
text_length = df['clean_text'].apply(len)
sorted_length = text_length.sort_values()[:-2000] # removed largest 2000 texts for sake of graphing (there are a few longer texts in the thousands with the largest being 8665)
plt.hist(sorted_length, bins=10)
plt.xlabel('Length of Text (in # of characters)')
plt.ylabel('Text Counts')
plt.title('Text Length Distribution')
plt.show()
