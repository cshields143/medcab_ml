import re
import cloudpickle

fh = open('data/word_vect.pkl', 'rb')
word_vect = cloudpickle.loads(fh.read())

if __name__ == '__main__':
	egq = 'indica giggly happy'
	print(word_vect.transform([egq]).todense().sum())