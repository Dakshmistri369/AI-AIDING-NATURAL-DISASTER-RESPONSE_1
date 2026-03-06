import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Sample Disaster Data
data = {
    'text': [
        "We are trapped in the house, water is rising!", 
        "Need medical assistance, broken leg here.",
        "The bridge on 5th street has collapsed.",
        "Send food and clean water to the shelter.",
        "Fire is spreading toward the residential area."
    ],
    'category': ['Flood', 'Medical', 'Infrastructure', 'Hunger', 'Fire']
}

df = pd.DataFrame(data)

# 2. Build a Machine Learning Pipeline
# CountVectorizer converts text to numbers; MultinomialNB is great for text classification
model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# 3. Train the model
model.fit(df['text'], df['category'])

# 4. Predict on a new emergency message
new_message = ["Help! My neighborhood is underwater and we are hungry."]
prediction = model.predict(new_message)

print(f"Alert Priority: {prediction[0]}")
