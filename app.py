import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import re

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin1')
data = data[['text', 'target']]  # Selecting relevant columns

# Inspect unique values in the 'target' column
print(data['target'].unique())  # Check for non-numeric values

# Convert 'target' column to numeric, replacing invalid values with NaN
data['target'] = pd.to_numeric(data['target'], errors='coerce')

# Drop rows where 'target' is NaN
data = data.dropna(subset=['target'])

# Convert 'target' to integer
data['target'] = data['target'].astype(int)

# Preprocess the text: Remove special characters, numbers, and convert to lowercase
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Apply preprocessing to the data
data['text'] = data['text'].apply(preprocess_text)

# Split data into features and labels
X = data['text']
y = data['target']

# Convert text data into numerical format using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')  # Use stop_words='english' to remove common stopwords
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model (optional)
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Save the trained model and vectorizer using pickle
with open('email_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully.")

# Function for user to classify emails
def classify_email(email_text):
    # Load the saved model and vectorizer
    with open('email_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    
    # Preprocess the input email text
    email_text = preprocess_text(email_text)
    
    # Transform the input email text
    email_vectorized = vectorizer.transform([email_text])
    
    # Predict the class (spam or not spam)
    prediction = model.predict(email_vectorized)
    
    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"

# User interaction
print("Welcome to the Email Classifier!")
while True:
    user_email = input("\nEnter the email content (or type 'exit' to quit): ")
    if user_email.lower() == 'exit':
        print("Exiting the Email Classifier. Goodbye!")
        break
    classification = classify_email(user_email)
    print(f"The email is classified as: {classification}")
