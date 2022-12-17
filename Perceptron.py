import numpy as np
from typing import Optional
class Perceptron(object):
    """Perceptron Classifier
    
    Parameters
    -------------
    eta: float
        Learning Rate between (0.0,1.0)
    n_iter: int
        Passes over training set
    random_state: int
        Randpö number generator seed for random weight
        initialization.
    
    Attributes
    -------------
    w_: id-array
        Weights after fitting
    errors_: list
        Number of misclassifications in each epoch
    """
    def __init__(self, eta: Optional[float] = 0.01, n_iter: Optional[int] = 50, random_state: Optional[int] = 1):
        self.eta= eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self,X,y):
        """ Fit training data.
        Parameters
        ------------
        X: {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of 
            examples and n_features is the number of features.
        y: array-like, shape = [n_features]
            Target values.
            
        Returns
        -----------
        self : object
        """
        
        rg = np.random.RandomState(self.random_state)
        self.w_ = rg.normal(loc = 0.0,scale = 0.01, 
                            size = 1 + X.shape[1])
        self.errors_ = []
        for _ in range (self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        """Return class label after running step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)