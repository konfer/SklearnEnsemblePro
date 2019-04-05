
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer

class CustomCategoryImputer(TransformerMixin):
	
	def __init__(self, cols = None, strategy = 'mean'):
		
		self.cols = cols
		self.strategy = strategy
	
	def transform(self, df):
		
		X = df.copy()
		impute = Imputer(strategy = self.strategy)
		
		for col in self.cols:
			X.loc[:, col] = impute.fit_transform(X[[col]])
		
		return X
	
	def fit(self, *_):
		return self