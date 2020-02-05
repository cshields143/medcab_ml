import re
import cloudpickle

fh = open('data/token_str.pkl', 'rb')
token_str = cloudpickle.loads(fh.read())



# run our model on a search query,
# return top 10 matching strains

#model = joblib.load('model.pkl')

#def find_match(q):
#	toks = mine_text(q)
#	probs = model.predict_proba([toks])[0]
#	lbld = list(zip(model.named_steps['clf'].classes_, probs))
#	lbld.sort(key=lambda x:x[1], reverse=True)
#	lbls = [l[0] for l in lbld]
#	return lbls[:10]



if __name__ == '__main__':
	print(open('data/cannabis_tokens.csv', 'r').read()[:100])