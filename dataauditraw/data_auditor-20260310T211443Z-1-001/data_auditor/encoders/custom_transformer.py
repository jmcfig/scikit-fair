from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin

#from encoders.ordinal import ProfilerOrdinalEncoder


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Create custom transformer for learning purposes.
       https://www.andrewvillazon.com/custom-scikit-learn-transformers/
    
    """

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # perform arbitrary transformation
        X["random_int"] = randint(0, 10, X.shape[0])
        return X
    

class MultiplyColumns(BaseEstimator, TransformerMixin):
    def __init__(self, by=1, columns=None):
        self.by = by
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        cols_to_transform = list(X.columns)

        if self.columns:
            cols_to_transform = self.columns

        X[cols_to_transform] = X[cols_to_transform] * self.by

        return X

    

    
