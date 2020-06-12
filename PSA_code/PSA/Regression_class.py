import numpy as np

class Regression:

    def __init__(self):
        pass

    def linearRegression_findBeta(self, X, Y, resolve_singularity=False):
        # ----- X --> rows: samples, columns: dimensions
        # ----- Y --> rows: samples, having one column
        # ----- linear regression:
        number_of_samples = X.shape[0]
        number_of_dimensions = X.shape[1]
        X = np.hstack((np.ones((number_of_samples,1)), X))
        if resolve_singularity == True:
            lambda_for_singularity = 0.0001
            beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lambda_for_singularity*np.eye(number_of_dimensions+1)), X.T), Y)
        else:
            beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
        return beta