import pandas as pd
import re
from sklearn.externals import joblib

# i hate everything
def dummy(x):
	return x





# run our model on a search query,
# return top 10 matching strains

model = joblib.load('model.pkl')

def find_match(q):
	toks = mine_text(q)
	probs = model.predict_proba([toks])[0]
	lbld = list(zip(model.named_steps['clf'].classes_, probs))
	lbld.sort(key=lambda x:x[1], reverse=True)
	lbls = [l[0] for l in lbld]
	return lbls[:10]



if __name__ == '__main__':
	print(find_match('indica giggly happy'))
	print('\n')
	print(find_match('frowny mean stupid'))