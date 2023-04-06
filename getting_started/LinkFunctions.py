from numpy import exp
import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64


def threshold_link(y: pd.Series, threshold: float = 0):
    '''
    Deterministic function to transform a variable into 0/1 class depending on fixed threshold
    Return class = 0 if Y is below or equal to threshold
    Retrun class = 1 if Y is above threshold

    Args:
        y: a pd.Series containing a variable that should be transformed into 0/1 class
        threshold: float that is used as cut-off value
    '''
    y_class = np.where(y <= threshold, 0, 1)
    return y_class


def logit_deterministic_link(y:pd.Series, probability_threshold: float = 0.5):
    '''
    Deterministic function to transform a variable into 0/1 class depending on probability threshold
    Return class = 0 if Y is below or equal to probability threshold
    Retrun class = 1 if Y is above threshold

    Args:
        y: a pd.Series containing a variable that should be transformed into 0/1 class
        threshold: float that is used as cut-off value
    '''
    probabilities = 1/(1 + np.exp(y))
    y_class = threshold_link(probabilities, probability_threshold)
    return y_class

def logit_random_link(y, seed=1):
    '''
    Random function to transform a variable into 0/1 class depending on probability.
    Probability is calculated by logit.
    Class is drawn from binomial distribution.

    Args:
        y: a pd.Series containing a variable that should be transformed into 0/1 class
        seed: seed for drawing from binomial
    '''
    g = Generator(PCG64(seed=seed))
    probabilities = 1/(1 + np.exp(y))
    y_class = [g.binomial(1, p, 1)[0] for p in probabilities]
    y_class = np.array(y_class)
    return(y_class)
