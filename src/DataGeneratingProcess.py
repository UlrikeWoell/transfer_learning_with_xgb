from numpy.random import Generator, PCG64
from numpy import exp
import pandas as pd
import LinkFunctions as lf
from typing import List


class Distribution:
    '''
    A short-cut to numpy's random sampling functionalities to generate features / covariates
    '''

    def __init__(self, seed: int, distribution_type: str, distribution_params: dict) -> None:
        '''
        Distribution_params:
        beta: {'a': 1 ,'b' : 2}
        uniform: {'low': 1, 'high': 3}
        normal: {'loc' : 0, 'scale' : 1}
        
        '''
        self.__admissible_distributions = ['beta', 'normal', 'uniform']
        self._rnd_generator = Generator(PCG64(seed=seed))

        self._distribution_type = distribution_type
        self._distribution_params = distribution_params
        self._sampling_function = self.set_sampling_function()

    @property
    def distribution_type(self):
        return self._distribution_type

    @distribution_type.setter
    def distribution_type(self, distribution_type: str):
        if distribution_type in self.__admissible_distributions:
            self._distribution_type = distribution_type
        else:
            raise ValueError(
                f'Unknown distribution type. Admissible values are {self.__admissible_distributions}')

    @property
    def distribution_params(self):
        return self._distribution_params

    @distribution_params.setter
    def distribution_params(self, params: dict):
        if self._distribution_type == 'normal':
            if 'loc' in params.keys() and 'scale' in params.keys():
                self._distribution_params = params
            else:
                raise KeyError('loc and scale must be given')

        elif self._distribution_type == 'beta':
            if 'a' in params.keys() and 'b' in params.keys():
                self._distribution_params = params
            else:
                raise KeyError('a and b must be given')

        elif self._distribution_type == 'uniform':
            if 'low' in params.keys() and 'high' in params.keys():
                self._distribution_params = params
            else:
                raise KeyError('low and high must be given')

        else:
            raise ValueError(
                f'Unknown distribution type. Admissible values are {self.__admissible_distributions}')

    @property
    def sampling_function(self):
        return self._sampling_function

    def set_sampling_function(self):
        available_distributions = {'beta': self._rnd_generator.beta,
                                   'normal': self._rnd_generator.normal,
                                   'uniform': self._rnd_generator.uniform}
        try:
            return available_distributions[self.distribution_type]

        except:
            KeyError(
                f'Could not find rnd_generator for {self.distribution_type} distribution')

    def sample(self, n):
        '''
        samples n from the given distribution
        '''
        return self.sampling_function(**self.distribution_params, size=n)




class Term:
    '''
    handles one covariate and its coefficient
    '''

    def __init__(self, coefficient: float, variable_name: str, variable_distribution: Distribution) -> None:
        self._coefficient = coefficient
        self._variable_name = variable_name
        self._variable_distribution = variable_distribution

    def print(self):
        print(f'{self.coefficient} {self._variable_name} ~ \
              {self._variable_distribution.distribution_type} \
              {self.variable_dristibution.distribution_params} ')

    @property
    def coefficient(self):
        return self._coefficient

    @coefficient.setter
    def coefficient(self, s: float):
        self._score = s

    @property
    def variable_name(self):
        return self._variable_name

    @variable_name.setter
    def variable_name(self, new_name: str):
        self._variable_name = new_name

    @property
    def variable_distribution(self):
        return self._variable_distribution

    @variable_distribution.setter
    def variable_dristibution(self, new_distribution: Distribution):
        self._variable_distribution = new_distribution

    def __str__(self) -> str:
        d_type = self.variable_distribution.distribution_type
        d_params = tuple(
            self.variable_distribution.distribution_params.values())
        return f"{self.coefficient}*{self.variable_name}  ~  {d_type} {d_params}"



class DataGeneratingProcess:
    '''
    DGP handles Coefficients and Covariates 
    Generates data and class variable
    '''

    def __init__(self, terms: List[Term], sample_size: int, link_fct: callable) -> None:
        '''
        Defines how covariates are combined to obtain the class variable

        Args:
        terms: a list of objects of class Term
        size: number of samples generate
        link_function: a function to transform Y into class
        
        Value:
        X: realizations of covariates, a pd.dataframe
        Y: class variable, a pd.dataframe
        link
        '''
        self.terms = terms
        self.sample_size = sample_size
        self.link_fct = link_fct
        self._feature_data = self.__gen_feature_data()
        self._target_data = self.__gen_target_data()
        self.linkedY = None
    


    @property 
    def feature_data(self):
        return self._feature_data
    
    def __gen_feature_data(self):
        x = {}
        for term in self.terms:
            var = term.variable_name
            data = term.variable_distribution.sample(self.sample_size)
            x[f'{var}'] = data
        return pd.DataFrame(x)
    
    @property
    def target_data(self):
        return self._target_data

    def __gen_target_data(self):
        y_raw = 0
        for term in self.terms:
            var = term.variable_name
            y_raw = y_raw + term.coefficient * self.feature_data[var]
        target = self.link_fct(y_raw)
        return target

    def describe(self):
        '''
        Prints the componentes of the DGP
        '''
        process = ''
        for t in self.terms:
            plus = '+' if t.coefficient >= 0 else ''
            process = process + plus + str(t.coefficient) + '*' + t.variable_name + ' '

        
        print(f'Process: {process} \n Variables: \n')
        for t in self.terms:
            t.print()
        print(f"\n Link function: {self.link_fct.__name__}")
