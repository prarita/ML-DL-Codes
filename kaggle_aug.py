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
