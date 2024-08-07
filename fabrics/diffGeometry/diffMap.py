import casadi as ca
import numpy as np

from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper, TorchFunctionWrapper
from fabrics.helpers.variables import Variables, TorchVariables
import torch
class TorchDifferentialMap:
    _vars: Variables
    # _J: ca.SX
    # _Jdot: ca.SX

    def __init__(self, phi, variables, **kwargs):
        self._vars = variables
        Jdot_sign = -1
        if 'Jdot_sign' in kwargs.keys():
            Jdot_sign = kwargs.get('Jdot_sign')
        q = self._vars.position_variable()
        qdot = self._vars.velocity_variable()
        if isinstance(phi, ca.SX) and isinstance(self._vars, Variables):
            J = ca.jacobian(phi,q)
            Jdot = Jdot_sign * ca.jacobian(ca.mtimes(J, qdot), q)
            self._caFunc = CasadiFunctionWrapper(
                "funs", self._vars, {"phi": phi, "J":J, "Jdot": Jdot}
            )
            inputs = list(self._caFunc._inputs.keys())
            self._q = inputs[0]
            self._qdot = inputs[1]
            self._vars = TorchVariables(position = self._q, velocity = self._qdot, parameter_variables=set(inputs[2:]))
            casadi = lambda **inputs: self._caFunc.evaluate(**inputs)
            self._phi = TorchFunctionWrapper(function=lambda **inputs: torch.tensor(casadi(**inputs)["phi"],dtype=torch.float64), variables=self._vars, \
                                             name="dm.phi", iscasadi=True)
            self._phi.set_name("phi")
            self._J = TorchFunctionWrapper(function=lambda **inputs: torch.tensor(casadi(**inputs)["J"],dtype=torch.float64), variables=self._vars,name="dm.J",iscasadi=True)
            self._J.set_name("J")
            self._Jdot = TorchFunctionWrapper(function=lambda **inputs: torch.tensor(casadi(**inputs)["Jdot"],dtype=torch.float64), variables=self._vars,name="dm._Jdot",iscasadi=True)
            self._Jdot.set_name("Jdot")
            self._Jdotqdot =self._Jdot @ self.qdot()   
            self._Jdotqdot.set_name("Jdotqdot")
        elif isinstance(phi, TorchFunctionWrapper) and isinstance(self._vars, TorchVariables):
            self._phi = phi
            self._phi.set_name("phi_limit")
            self._J = phi.grad(q)
            self._J.set_name("J_limit")
            self._Jdot =TorchFunctionWrapper(function=lambda **inputs: torch.zeros(7),variables=self._vars)
                # Jdot_sign * (self._J @ self.qdot()).grad(q)
            self._Jdot.set_name("Jdot_limit")
            self._Jdotqdot = self._Jdot @ self.qdot()
            self._Jdotqdot.set_name("Jdotqdot_limit")
        elif isinstance(phi, str):
            self._phi = phi
            # self._J = J
            #etc
        else:
            raise Exception("type error!")
            
        # self.f_phi = ca.Function('f_phi', [q], [self._phi])
        # self.f_J = ca.Function('f_J', [q, qdot], [self._J])
        # self.f_Jdot = ca.Function('f_Jdot', [q, qdot], [self._Jdot])
        
    def params(self) -> dict:
        return self._vars.parameters()
    
    def q(self):
        func = lambda **kwargs : kwargs[self._vars._position]
        return TorchFunctionWrapper(function=func, variables = self._vars, name="q in DiffMap")

    def qdot(self):
        func = lambda **kwargs : kwargs[self._vars._velocity]
        return TorchFunctionWrapper(function=func, variables = self._vars, name= "qdot in DiffMap")
    


class DifferentialMap:
    _vars: Variables
    _J: ca.SX
    _Jdot: ca.SX

    def __init__(self, phi: ca.SX, variables: Variables, **kwargs):
        assert isinstance(phi, ca.SX)
        assert isinstance(variables, Variables)
        self._vars = variables
        Jdot_sign = -1
        if 'Jdot_sign' in kwargs.keys():
            Jdot_sign = kwargs.get('Jdot_sign')
        self._vars.verify()
        self._phi = phi
        q = self._vars.position_variable()
        qdot = self._vars.velocity_variable()
        self._J = ca.jacobian(phi, q)
        self._phidot = ca.mtimes(self._J, qdot)
        self._Jdot = Jdot_sign * ca.jacobian(ca.mtimes(self._J, qdot), q)
        # self.f_phi = ca.Function('f_phi', [q], [self._phi])
        # self.f_J = ca.Function('f_J', [q, qdot], [self._J])
        # self.f_Jdot = ca.Function('f_Jdot', [q, qdot], [self._Jdot])
    def Jdotqdot(self) -> ca.SX:
        return ca.mtimes(self._Jdot, self.qdot())

    def phidot(self) -> ca.SX:
        return ca.mtimes(self._J, self.qdot())

    def concretize(self) -> None:
        self._funs = CasadiFunctionWrapper(
            "funs", self._vars, {"phi": self._phi, "phidot": self._phidot, "J":self._J, "Jdot": self._Jdot}
        )
    def params(self) -> dict:
        return self._vars.parameters()

    def state_variables(self) -> dict:
        return self._vars.state_variables()

    def forward(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        x = evaluations['phi']
        xdot = evaluations['phidot']
        J = evaluations['J']
        Jdot = evaluations['Jdot']
        return x, xdot, J, Jdot

    def q(self):
        return self._vars.position_variable()

    def qdot(self):
        return self._vars.velocity_variable()

class DynamicDifferentialMap(DifferentialMap):
    _phi_dot: ca.SX
    _Jdotqdot: ca.SX

    def __init__(self, variables: Variables, ref_names=['x_ref', 'xdot_ref', 'xddot_ref'], **kwargs):
        self._x_ref_name = ref_names[0]
        self._xdot_ref_name = ref_names[1]
        self._xddot_ref_name = ref_names[2]
        phi = variables.position_variable() - variables.parameter_by_name(self._x_ref_name)
        self._phi_dot = variables.velocity_variable() - variables.parameter_by_name(self._xdot_ref_name)

        super().__init__(phi, variables, **kwargs)

    def x_ref(self) -> ca.SX:
        return self._vars.parameter_by_name(self._x_ref_name)

    def xdot_ref(self) -> ca.SX:
        return self._vars.parameter_by_name(self._xdot_ref_name)

    def xddot_ref(self) -> ca.SX:
        return self._vars.parameter_by_name(self._xddot_ref_name)

    def ref_names(self) -> list:
        return [self._x_ref_name, self._xdot_ref_name, self._xddot_ref_name]

    def phidot(self) -> ca.SX:
        return self._phi_dot

    def concretize(self) -> None:
        self._funs = CasadiFunctionWrapper(
            "funs", self._vars, {"x_rel": self._phi, "xdot_rel": self._phi_dot}
        )

    def forward(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        x = evaluations['x_rel']
        xdot = evaluations['xdot_rel']
        return x, xdot

class ExplicitDifferentialMap(DifferentialMap):
    """Explicit differential map for which the gradients can be computed at runtime.

    This class is a special differential map for which the Jacobian matrices
    can be set numerically at runtime.
    """
    def __init__(self, phi: ca.SX, variables: Variables, **kwargs):
        super().__init__(phi, variables, **kwargs)
        try:
            self._J = kwargs.get("J")
            self._Jdot = kwargs.get("Jdot")
        except Exception as e:
            raise Exception("J and Jdot not defined for ExplicitDifferentialMap")

