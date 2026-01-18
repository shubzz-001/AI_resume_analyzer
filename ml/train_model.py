import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Loading Dataset
df = pd.read_csv("data/job_descriptions.csv")

x = df["job_description"]
y = df["job_title"]

# Creating A Pipeline
pipeline = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=3000,ngram_range=(1,2))),
    ("clf",LogisticRegression(max_iter=1000))
])

# Train Model
pipeline.fit(x,y)

# Model Saving
with open("ml/job_role_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Job role prediction model trained and saved")