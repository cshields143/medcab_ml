import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import cloudpickle
import os

this_path = os.path.dirname(__file__)
fn1 = os.path.join(this_path, '../data/cannabis_tokens.csv')
fn2 = os.path.join(this_path, '../data/word_vect.pkl')

# from make/data.py
df = pd.read_csv(fn1)

def dummy(x): return x
vect = TfidfVectorizer(preprocessor=dummy, tokenizer=dummy)
vect.fit(df['tokens'])

pkl = cloudpickle.dumps(vect)
with open(fn2, 'wb') as fh:
	fh.write(pkl)