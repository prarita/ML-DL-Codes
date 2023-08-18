
#using tfidf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
train_data = pd.read_csv('summaries_train.csv')
test_data = pd.read_csv('summaries_test.csv')
prompts_train = pd.read_csv('prompts_train.csv')

# Merge training data with prompts data
train_data = train_data.merge(prompts_train, on='prompt_id', how='left')

# Preprocessing - TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features
X_text = vectorizer.fit_transform(train_data['text'])
X_text_test = vectorizer.transform(test_data['text'])

# Feature Engineering - Prompt-related Features
train_data['prompt_length'] = train_data['prompt_text'].apply(len)
test_data['prompt_length'] = test_data['prompt_text'].apply(len)

# Combine all features
X_prompt_features = train_data[['prompt_length']].values
X_all = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
X_all = pd.concat([X_all, pd.DataFrame(X_prompt_features, columns=['prompt_length'])], axis=1)

# Define targets
y_content = train_data['content']
y_wording = train_data['wording']

# Split the data
X_train, X_val, y_train_content, y_val_content, y_train_wording, y_val_wording = train_test_split(
    X_all, y_content, y_wording, test_size=0.2, random_state=42
)

# Model - Linear Regression (you can use more complex models)
model_content = LinearRegression()
model_wording = LinearRegression()

# Train the models
model_content.fit(X_train, y_train_content)
model_wording.fit(X_train, y_train_wording)

# Predict on validation data
pred_content = model_content.predict(X_val)
pred_wording = model_wording.predict(X_val)

# Evaluate using Mean Squared Error
mse_content = mean_squared_error(y_val_content, pred_content)
mse_wording = mean_squared_error(y_val_wording, pred_wording)

print(f"Content MSE: {mse_content}")
print(f"Wording MSE: {mse_wording}")




#embeddings_Word2Vec
To calculate the MCRMSE and create the submission file using NLP embeddings, you can use pre-trained word embeddings (such as Word2Vec, GloVe, or fastText) to convert the text data into numerical representations. In this example, I'll demonstrate using Word2Vec embeddings and then proceed with calculating MCRMSE and generating the submission file.

Here's the modified code:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from gensim.models import Word2Vec
from tqdm import tqdm

# Load the datasets
train_data = pd.read_csv('summaries_train.csv')
test_data = pd.read_csv('summaries_test.csv')
prompts_train = pd.read_csv('prompts_train.csv')

# Merge training data with prompts data
train_data = train_data.merge(prompts_train, on='prompt_id', how='left')

# Load pre-trained Word2Vec model
w2v_model = Word2Vec.load('path_to_word2vec_model')

# Function to convert text to embeddings
def text_to_embeddings(text):
    words = text.split()
    embeddings = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

# Convert text data to embeddings
X_text_embeddings = np.array([text_to_embeddings(text) for text in tqdm(train_data['text'])])
X_text_embeddings_test = np.array([text_to_embeddings(text) for text in tqdm(test_data['text'])])

# Feature Engineering - Prompt-related Features
train_data['prompt_length'] = train_data['prompt_text'].apply(len)
test_data['prompt_length'] = test_data['prompt_text'].apply(len)

# Combine all features
X_prompt_features = train_data[['prompt_length']].values
X_all = np.hstack((X_text_embeddings, X_prompt_features))

# Define targets
y_content = train_data['content']
y_wording = train_data['wording']

# Split the data
X_train, X_val, y_train_content, y_val_content, y_train_wording, y_val_wording = train_test_split(
    X_all, y_content, y_wording, test_size=0.2, random_state=42
)

# Train the model using Random Forest or Gradient Boosting Regression as before

# ...

# Calculate MCRMSE
mse_content = mean_squared_error(y_val_content, pred_content_rf)
mse_wording = mean_squared_error(y_val_wording, pred_wording_rf)

mcrmse = np.sqrt((mse_content + mse_wording) / 2)

print(f"MCRMSE: {mcrmse}")

# Generate submission file
X_test = np.hstack((X_text_embeddings_test, test_data[['prompt_length']].values))
pred_content_test = model_content_rf.predict(X_test)
pred_wording_test = model_wording_rf.predict(X_test)

submission_df = pd.DataFrame({
    'student_id': test_data['student_id'],
    'content': pred_content_test,
    'wording': pred_wording_test
})

submission_df.to_csv('submission.csv', index=False)
```

 replace `'path_to_word2vec_model'` with the actual path to your trained Word2Vec model.
