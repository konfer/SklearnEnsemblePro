import pandas as pd


from sklearn.base import TransformerMixin

class CustomDummifierTrans(TransformerMixin):
	
	def __init__(self, cols = None):
		self.cols = cols
	
	def transform(self, X):
		return pd.get_dummies(X, columns = self.cols)
	
	def fit(self, *_):
		return self