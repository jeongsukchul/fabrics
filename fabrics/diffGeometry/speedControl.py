import casadi as ca

from fabrics.diffGeometry.diffMap import DifferentialMap, TorchDifferentialMap
from fabrics.diffGeometry.energy import Lagrangian, TorchLagrangian
from fabrics.helpers.functions import parse_symbolic_input
from fabrics.helpers.casadiFunctionWrapper import TorchFunctionWrapper
from fabrics.planner.configuration_classes import damperBeta, damperEta
from fabrics.helpers.variables import Variables, TorchVariables


import torch

class TorchDamper:
    def __init__(
        self, beta_expression: damperBeta, eta_expression: damperEta, x: str , dm: TorchDifferentialMap, lagrangian_execution: TorchLagrangian
    ):
        self._x = x
        self._dm = dm
        self._a_le = "a_le"
        self._a_ex = "a_ex" 
        self._beta = TorchFunctionWrapper(expression=beta_expression, variables=self._dm._vars, ex_input = [x, self._a_ex,self._a_le], name="beta")
        self._eta = TorchFunctionWrapper(expression=eta_expression, variables=self._dm._vars, ex_input = [self._dm._vars._velocity], name="eta")

    def substitute_beta(self, a_ex_fun, a_le_fun):
        def subst_beta(**kwargs):
            upper_kwargs = kwargs
            upper_kwargs[self._x] = self._dm._phi(**kwargs)
            upper_kwargs[self._a_ex] = a_ex_fun(**kwargs)
            upper_kwargs[self._a_le] = a_le_fun(**kwargs)
            # print("upper_kwargs", upper_kwargs)
            return self._beta(**upper_kwargs)
        return TorchFunctionWrapper(function=subst_beta, variables=self._dm._vars, name="beta_subst")
    def substitute_eta(self):
        return self._eta
class Damper:
    def __init__(
        self, beta_expression: str, eta_expression: str, x: ca.SX, dm: DifferentialMap, lagrangian_execution: ca.SX
    ):
        assert isinstance(beta_expression, str)
        assert isinstance(eta_expression, str)
        assert isinstance(x, ca.SX)
        assert isinstance(dm, DifferentialMap)
        assert isinstance(lagrangian_execution, ca.SX)
        self._x = x
        self._dm = dm
        self._symbolic_parameters = {}

        self.parse_beta_expression(beta_expression)
        self.parse_eta_expression(eta_expression, lagrangian_execution)

    def parse_beta_expression(self, beta_expression):

        beta_parameters, self._beta = parse_symbolic_input(beta_expression, self._x, None, 'damper')
        print("beta param", beta_parameters)
        print("beta", self._beta)
        if 'a_ex_damper' in beta_parameters:
            self._a_ex = beta_parameters['a_ex_damper']
            self._a_le = beta_parameters['a_le_damper']
            del(beta_parameters['a_ex_damper'])
            del(beta_parameters['a_le_damper'])
            self._constant_beta_expression = False
        else:
            self._constant_beta_expression = True
        self._symbolic_parameters.update(beta_parameters)
    def parse_eta_expression(self, eta_expression, lagrangian_execution):
        qdot = ca.vcat(ca.symvar(lagrangian_execution))
        eta_parameters, eta_raw = parse_symbolic_input(eta_expression, None, qdot, 'damper')
        print("eta param", eta_parameters)
        print("eta", eta_raw)
        if 'ex_lag_damper' in eta_parameters:
            ex_lag = eta_parameters['ex_lag_damper']
            self._eta = ca.substitute(eta_raw, ex_lag, lagrangian_execution)
            del(eta_parameters['ex_lag_damper'])
        else:
            self._eta = eta_raw
        self._symbolic_parameters.update(eta_parameters)
    def symbolic_parameters(self):
        return self._symbolic_parameters

    def substitute_beta(self, a_ex_fun, a_le_fun):
        if not self._constant_beta_expression:
            beta_subst = ca.substitute(self._beta, self._a_ex, a_ex_fun)
            beta_subst2 = ca.substitute(beta_subst, self._a_le, a_le_fun)
            beta_subst3 = ca.substitute(beta_subst2, self._x, self._dm._phi)
            return beta_subst3
        else:
            beta_subst = ca.substitute(self._beta, self._x, self._dm._phi)
            return beta_subst

    def substitute_eta(self):
        return self._eta

class Interpolator:
    def __init__(self, eta: ca.SX, lex: Lagrangian, lex_d: Lagrangian):
        assert isinstance(eta, ca.SX)
        assert isinstance(lex, Lagrangian)
        assert isinstance(lex_d, Lagrangian)
        self._eta = eta
        self._lex = lex
        self._lex_d = lex_d
