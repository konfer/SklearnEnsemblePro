from sklearn.base import TransformerMixin

class CustomEncodingTrans(TransformerMixin):
	
	def __init__(self, col, ordering = None):
		self.ordering = ordering
		self.col = col
	
	def transform(self, df):
		X = df.copy()
		X.loc[:, self.col] = X.loc[:, self.col].map(lambda x: self.ordering.index(x))
		return X
	
	def fit(self, *_):
		return self