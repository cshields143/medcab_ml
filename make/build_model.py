import pandas as pd
from api import mine_text, dummy
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# i hate everything
def dummy(x):
	return x

# from build_data.py
df = pd.read_csv('tokenized.csv')

# from build_vect.py
vec = joblib.load('word_vect.pkl')

clf = LogisticRegression()
pipe = Pipeline([
  ('vec', vec),
  ('clf', clf)
])
pipe.fit(df['tokens'], df['Strain'])

joblib.dump(pipe, 'model.pkl')