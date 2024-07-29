from typing import Union
import casadi as ca
import numpy as np
from copy import deepcopy
import torch

class ParameterNotFoundError(Exception):
    pass

class Variables(object):
    def __init__(self, state_variables=None, parameters=None, parameters_values=None):
        if state_variables is None:
            state_variables = {}
        if parameters is None:
            parameters = {}
        if parameters_values is None:
            parameters_values = {}
        self._state_variables = state_variables
        self._parameters = parameters
        self._parameters_values = parameters_values
        # if len(state_variables) > 0:
        self._state_variable_names = list(state_variables.keys())
        # if len(parameters) > 0:
        self._parameter_names = list(parameters.keys())

    def state_variables(self):
        return self._state_variables

    def add_state_variable(self, name, value):
        self._state_variables[name] = value

    def parameters(self) -> dict:
        return self._parameters

    def parameters_values(self) -> dict:
        return self._parameters_values

    def add_parameter(self, name: str, value: ca.SX) -> None:
        self._parameters[name] = value

    def add_parameter_value(self, name: str, value: Union[float, np.ndarray]) -> None:
        if name not in self._parameters:
            raise ParameterNotFoundError(f"Parameter {name} not in parameters")
        self._parameters_values[name] = value

    def add_parameters(self, parameter_dict: dict) -> None:
        self._parameters.update(parameter_dict)

    def add_parameters_values(self, parameter_dict: dict) -> None:
        for parameter_name, parameter_value in parameter_dict.items():
            self.add_parameter_value(parameter_name, parameter_value)

    def set_parameters(self, parameters):
        self._parameters = parameters

    def variable_by_name(self, name: str) -> ca.SX:
        return self._state_variables[name]

    def parameter_by_name(self, name: str) -> ca.SX:
        try:
            return self._parameters[name]
        except KeyError as key_error:
            raise ParameterNotFoundError(f"Parameter {name} not in variables, available ones are {self._parameters.keys()}")

    def position_variable(self) -> ca.SX:
        return self.variable_by_name(self._state_variable_names[0])

    def velocity_variable(self) -> ca.SX:
        return self.variable_by_name(self._state_variable_names[1])

    def verify(self):
        for key in self._state_variables:
            assert isinstance(key, str)
            assert isinstance(self._state_variables[key], ca.SX)
        for key in self._parameters:
            assert isinstance(key, str)
            assert isinstance(self._parameters[key], ca.SX)

    def asDict(self):
        joinedDict = {}
        joinedDict.update(self._state_variables)
        joinedDict.update(self._parameters)
        return joinedDict

    def __add__(self, b):
        joined_state_variables = deepcopy(self._state_variables)
        for key, value in b.state_variables().items():
            if key in joined_state_variables:
                if ca.is_equal(joined_state_variables[key], value):
                    continue
                else:
                    new_key = key
                    counter = 1
                    while new_key in joined_state_variables.keys():
                        new_key = key + "_" + str(counter)
                        counter += 1
                    joined_state_variables[new_key] = value
            else:
                joined_state_variables[key] = value
        joined_parameters = deepcopy(self._parameters)
        for key, value in b.parameters().items():
            if key in joined_parameters:
                if ca.is_equal(joined_parameters[key], value):
                    continue
                else:
                    new_key = key
                    counter = 1
                    while new_key in joined_parameters.keys():
                        new_key = key + "_" + str(counter)
                        counter += 1
                    joined_parameters[new_key] = value
            else:
                joined_parameters[key] = value
        joined_parameters_values = {**self.parameters_values(), **b.parameters_values()}
        return Variables(
            state_variables=joined_state_variables,
            parameters=joined_parameters,
            parameters_values=joined_parameters_values,
        )

    def len(self):
        return len(self._parameters.values()) + len(
            self._state_variables.values()
        )

    def __repr__(self):
        return self.__str__();

    def __str__(self):
        return (
            "State variables: "
            + self._state_variables.__str__()
            + "| parameters : "
            + self._parameters.__str__()
        )

    def toTorch(self):
        return TorchVariables(position= self._state_variable_names[0], velocity = self._state_variable_names[1], parameter_variables = set(self._parameter_names))

class TorchVariables: # it gives only function variables name[key]  i will not use parameter for now
    def __init__(self,  position='q', velocity ='qdot', parameter_variables = None):
        # Initialize the superclass (Variables)
        # Additional initialization for TorchVariables
        self._position = position
        self._velocity = velocity
        if parameter_variables is None:
            parameter_variables = set()
        self._parameter_variables = parameter_variables
        
    def __repr__(self):
        return f"TorchVariables(position={self._position},velocity={self._velocity}, parameter={self._parameter_variables})"

    def __add__(self, b):
        if self._position != b._position or self._velocity != b._velocity:
            raise Exception("Two Variables are in different space")
        joined_parameter_variables = self._parameter_variables | b._parameter_variables
        return TorchVariables(
            position=self._position,
            velocity=self._velocity,
            parameter_variables=joined_parameter_variables,
        )

    def asArray(self):
        return [self._position, self._velocity]+list(self._parameter_variables)
    
    def position_velocity_variables(self) -> set:
        return [self._position, self._velocity]

    def position_variable(self) -> str:
        return self._position

    def velocity_variable(self) -> str:
        return self._velocity
    
    def add_parameter(self, name: str):   
        self._parameter_variables = self._parameter_variables | {name}
    def add_parameters(self, _set: set):   
        self._parameter_variables = self._parameter_variables | _set
