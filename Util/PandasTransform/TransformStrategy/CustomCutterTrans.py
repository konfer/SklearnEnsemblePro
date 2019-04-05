from sklearn.base import TransformerMixin

class CustomCutter(TransformerMixin):
	
	def __init__(self, col, bins, labels = False):
		self.labels = labels
		self.col = col
		self.bins = bins
	
	def transform(self, df):
		X = df.copy()
		X.loc[:, self.col] = pd.cut(X.loc[:, self.col], bins = self.bins, labels = self.labels)
		return X
	
	def fit(self, *_):
		return self