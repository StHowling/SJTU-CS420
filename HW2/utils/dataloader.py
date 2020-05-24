from sklearn import datasets
import numpy as np

def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y

def genTwoMoons(n_samples=1000):
    X,y = datasets.make_moons(n_samples, noise=0.05)
    return X,y

def genTwoGaussians(n_samples=1000):
    np.random.seed(0)
    C = np.array([[0.5, -0.1], [.7, 1.4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-2, 2]),
         .5 * np.random.randn(n_samples, 2) + np.array([-4, -1])]
    return X,C
