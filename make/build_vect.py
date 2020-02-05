import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# i hate everything
def dummy(x):
	return x

# from build_data.py
df = pd.read_csv('tokenized.csv')

vect = TfidfVectorizer(preprocessor=dummy, tokenizer=dummy)
vect.fit(df['tokens'])

#joblib.dump(vect, 'word_vect.pkl')
print(pd.DataFrame(vect.transform([['frown', 'mean', 'stupid']]).todense()).T.sum())