import pandas as pd
import cloudpickle
import os

this_path = os.path.dirname(__file__)
fn = os.path.join(this_path, '../data/token_str.pkl')
fh = open(fn, 'rb')
token_str = cloudpickle.loads(fh.read())

def preprocess_table(filename):
	'''
	Open a file of data & clean/normalize it to our schema
	'''

	df = pd.read_csv(filename)

	# drop duplicate rows
	df = df.drop_duplicates()

	# fill missing values
	df['Description'] = df['Description'].fillna('None')
	df['Flavor'] = df['Flavor'].fillna('None')

	# we're assuming the table of data lacks symptom info;
	# this part of the code might be taken out later
	df['Symptoms'] = ['None'] * df.shape[0]

	return df

def create_fulltext(df):
	'''
	Given a table that matches our schema,
	create a table that only has name of strain
	and all of the fields combined into one string of text
	'''
	fulltext = []
	for i in range(df.shape[0]):
		fulltext.append(mine_row(df.iloc[i]))
	df['tokens'] = fulltext
	return df[['Strain', 'tokens']]

def mine_row(r):
	'''
	Given a row from a dataframe, mine each field
	for tokens to add to "fulltext"
	'''
	tok_buk = list()

	# schema has 7 fields; mine each of them
	for i in range(7):
		for _ in token_str(r[i]):
			tok_buk.append(_.lower())

	# turn our set into a list before returning
	return tok_buk

fn1 = os.path.join(this_path, '../data/cannabis.csv')
fn2 = os.path.join(this_path, '../data/cannabis_tokens.csv')
df = create_fulltext(preprocess_table(fn1))
df.to_csv(fn2, index=False)