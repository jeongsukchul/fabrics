import casadi as ca
import numpy as np
from copy import deepcopy

from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap, TorchDifferentialMap
from fabrics.helpers.variables import TorchVariables, Variables
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper, TorchFunctionWrapper





class Geometry:
    """description"""

    def __init__(self, **kwargs):
        self._x_ref_name = "x_ref"
        self._xdot_ref_name = "xdot_ref"
        if 'x' in kwargs:
            h = kwargs.get("h")
            self._vars = Variables(state_variables={"x": kwargs.get('x'), "xdot": kwargs.get('xdot')})
        if 'var' in kwargs:
            h = kwargs.get("h")
            self._vars = kwargs.get('var')
        elif 's' in kwargs:
            s = kwargs.get("s")
            h = s.h()
            self._vars = s._vars
        self._refTrajs = []
        if 'refTrajs' in kwargs:
            self._refTrajs = kwargs.get('refTrajs')

        self._h = h

    def x(self):
        return self._vars.position_variable()

    def xdot(self):
        return self._vars.velocity_variable()

    def __add__(self, b):
        assert isinstance(b, Geometry)
        # TODO: checkCompatibility()  <24-07-21, mspahn> #
        var = self._vars + b._vars
        return Geometry(h=self._h + b._h, var=var)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        h_pulled = ca.mtimes(ca.pinv(dm._J), self._h + dm.Jdotqdot())
        h_pulled_subst_x = ca.substitute(h_pulled, self.x(), dm._phi)
        h_pulled_subst_x_xdot = ca.substitute(h_pulled_subst_x, self.xdot(), dm.phidot())
        new_state_variables = dm.state_variables()
        new_parameters = {}
        new_parameters.update(self._vars.parameters())
        new_parameters.update(dm.params())
        new_vars = Variables(state_variables=new_state_variables, parameters=new_parameters)
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        return Geometry(h=h_pulled_subst_x_xdot, var=new_vars, refTrajs=refTrajs)

    def dynamic_pull(self, dm: DynamicDifferentialMap):
        h_pulled = self._h - dm.xddot_ref()
        h_pulled_subst_x = ca.substitute(h_pulled, self.x(), dm._phi)
        h_pulled_subst_x_xdot = ca.substitute(h_pulled_subst_x, self.xdot(), dm.phidot())
        return Geometry(h=h_pulled_subst_x_xdot, var=dm._vars)

    def concretize(self):
        self._xddot = -self._h
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        self._funs = CasadiFunctionWrapper(
            "funs", var, {"h": self._h, "xddot": self._xddot}
        )

    def evaluate(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        h_eval = evaluations['h']
        xddot_eval = evaluations['xddot']
        return [h_eval, xddot_eval]

    def testHomogeneousDegree2(self):
        x = np.random.rand(self.x().size()[0])
        xdot = np.random.rand(self.xdot().size()[0])
        alpha = 2.0
        xdot2 = alpha * xdot
        h, _ = self.evaluate(x=x, xdot=xdot)
        h2, _ = self.evaluate(x=x, xdot=xdot2)
        return h * alpha ** 2 == h2


class TorchGeometry(object):
    def __init__(self, **kwargs):
        # if 'x' in kwargs:
        #     h = kwargs.get("h")
        #     self._vars = Variables(state_variables={"x": kwargs.get('x'), "xdot": kwargs.get('xdot')})
        if 'var' in kwargs:
            h = kwargs.get("h")
            self._vars = kwargs.get('var')
        elif 's' in kwargs:
            s = kwargs.get("s")
            h = s._h
            self._vars = s._vars
        self._h = h
        self._x = self._vars.position_variable()
        self._xdot = self._vars.velocity_variable()

    def __add__(self, b):
        assert isinstance(b, TorchGeometry)
        # TODO: checkCompatibility()  <24-07-21, mspahn> #
        var = self._vars + b._vars
        return TorchGeometry(h=self._h + b._h, var=var)
    def pull(self, dm: TorchDifferentialMap):
        assert isinstance(dm, TorchDifferentialMap)
        h_pulled = dm._J.pinv() @ (self._h.lowerLeaf(dm) + dm._Jdotqdot)
        new_vars = TorchVariables(position = dm._vars._position, velocity= dm._vars._velocity, parameter_variables=dm._vars._parameter_variables | self._vars._parameter_variables)

        return TorchGeometry(h=h_pulled, var=new_vars)
    
    def evaluate(self, x, xdot):
        return self._h(x,xdot)
    
    def x(self):
        func = lambda **kwargs : kwargs[self._x]
        return TorchFunctionWrapper(function=func, variables = self._vars, name = "x in Geom")

    def xdot(self):
        func = lambda **kwargs : kwargs[self._xdot]
        return TorchFunctionWrapper(function=func, variables = self._vars, name = "xdot in Geom")