import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, ClassLabel
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt 

def tokenize_function(examples):
    return tokenizer(examples['clean_text'], padding='max_length', truncation=True, max_length=128)

# load cleaned data
df = pd.read_pickle('./ML models/cleaned_data.pkl')

# sentiment encoding 
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment'])

# convert dataframe to hugging face dataset 
dataset = Dataset.from_pandas(df[['clean_text', 'label']])

# convert target column to ClassLabel type
class_names = list(label_encoder.classes_)
class_label = ClassLabel(names=class_names)
dataset = dataset.cast_column('label', class_label)

# split dataset into training and test sets
train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column='label', seed=3)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# load pre-trained BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# tokenize the data
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# define training arguments
training_args = TrainingArguments(
    output_dir='./ML models/BERT/results',     
    evaluation_strategy='epoch',
    save_strategy='epoch',  
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,   
    num_train_epochs=3,         
    weight_decay=0.01,               
    logging_dir='./ML models/BERT/logs',             
    logging_steps=10,
    save_total_limit=2,               
    load_best_model_at_end=True         
)

# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=None
)

# train model
trainer.train()

# examine results 
eval_results = trainer.evaluate()
print(f'Evaluation results: {eval_results}')

# generate predictions for the test set 
preds_output = trainer.predict(test_dataset)
y_pred = torch.argmax(torch.tensor(preds_output.predictions), axis=1).numpy()
y = test_dataset['label']

# examine accuracy score, classification report, and confusion matrix
accuracy = accuracy_score(y, y_pred)
print(f'Accuracy Score: {accuracy}')
report = classification_report(y, y_pred, target_names=label_encoder.classes_)
print(f'Classification Report:\n {report}')
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# save tokenizer and model
tokenizer.save_pretrained('./ML models/BERT/')
joblib.dump(label_encoder, './ML models/BERT/label_encoder.pkl')
model.save_pretrained('./ML models/BERT/')
