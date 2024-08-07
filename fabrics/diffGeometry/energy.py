import casadi as ca
import numpy as np
import logging

from copy import deepcopy

from fabrics.diffGeometry.spec import Spec, checkCompatability, TorchSpec
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap, TorchDifferentialMap

from fabrics.helpers.functions import joinRefTrajs
from fabrics.helpers.variables import Variables, TorchVariables
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper, TorchFunctionWrapper

from fabrics.helpers.constants import eps
import torch
import functorch


class LagrangianException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


class TorchLagrangian(object):
    """only for basic Euler Lagrangian is implemented"""
    def __init__(self, l, **kwargs):
        # assert isinstance(l, baseEnergy)
        self._l = l
        vars = kwargs.get('var')
        assert isinstance(vars, TorchVariables)
        self._vars = vars

        self._x = self._vars.position_variable()
        self._xdot = self._vars.velocity_variable()

        if "isLimit" in kwargs:
            self._isLimit = kwargs.get("isLimit")
        else:
            self._isLimit = False
        if 'spec' in kwargs:
            # self._H = kwargs.get('hamiltonian')
            self._S = kwargs.get('spec')
        else:
            self.applyEulerLagrange()


    def x(self):
        func = lambda **kwargs : kwargs[self._x]
        return TorchFunctionWrapper(function=func, variables = self._vars, name="x in Lagrangian")

    def xdot(self):
        func = lambda **kwargs : kwargs[self._xdot]
        return TorchFunctionWrapper(function=func, variables = self._vars, name="xdot in Lagrangian")
    
    def applyEulerLagrange(self):

        # Compute gradients and Jacobians using functorch
        if self._isLimit:
            dL_dxdot = self._l.grad_elementwise(self._xdot)
            dL_dxdot.set_name("dL_dxdot_limit")
            dL_dx = self._l.grad_elementwise(self._x)
            dL_dx.set_name("dL_dx_limit")
            d2L_dxdxdot = dL_dx.grad_elementwise(self._xdot)
            d2L_dxdot2 = dL_dxdot.grad_elementwise(self._xdot)
            F = d2L_dxdxdot.diag()
            F.set_name("F:d2L_dxdxdot_limit")
            f_e = -dL_dx
            f_e.set_name("f_e_limit")
            M = d2L_dxdot2.diag()
            M.set_name("M:d2L_dxdot2_limit")
            f = F @ self.xdot() + f_e
            f.set_name("lagrange f_limit")
            self._dL_dxdot = dL_dxdot
            # self._H = dL_dxdot @ self.xdot() - self._l 
            # self._H.set_name("limit hamiltonian")
            self._S = TorchSpec(M, f=f, var=self._vars)
        else:
            dL_dxdot = self._l.grad(self._xdot)
            dL_dxdot.set_name("dL_dxdot")
            dL_dx = self._l.grad(self._x)
            dL_dx.set_name("dL_dx")
            d2L_dxdxdot = dL_dx.grad(self._xdot)
            d2L_dxdot2 = dL_dxdot.grad(self._xdot)
            F = d2L_dxdxdot
            F.set_name("F:d2L_dxdxdot")
            f_e = -dL_dx
            
            M = d2L_dxdot2
            M.set_name("M:d2L_dxdot2")
            f = F.transpose() @ self.xdot() + f_e
            f.set_name("lagrange f")

            self._dL_dxdot = dL_dxdot
            # self._H = dL_dxdot @ self.xdot() - self._l 
            # self._H.set_name("hamiltonian")
            self._S = TorchSpec(M, f=f, var=self._vars)

    
    def pull(self, dm: TorchDifferentialMap, isLimit:bool = False):
        assert isinstance(dm, TorchDifferentialMap)
        if isLimit:
            l = self._l.lowerLeaf(dm).sum()
            l.set_name("l_pulled_limit")
            # H = self._H.lowerLeaf(dm)
            # H.set_name("H_pulled_limit")
            M = self._S._M.lowerLeaf(dm)
            M.set_name("lag M_pulled_limit")
            f = self._S._f.lowerLeaf(dm)
            f.set_name("lag f_pulled_limit")
            new_vars = TorchVariables(position = dm._vars._position, velocity= dm._vars._velocity, parameter_variables=dm._vars._parameter_variables | self._vars._parameter_variables)
            S_pulled = TorchSpec(M, f=f, var=new_vars)
            # return TorchLagrangian(l, spec = S_pulled, hamiltonian=H, var=new_vars)
            return TorchLagrangian(l, spec = S_pulled, var=new_vars)
        # else:
        #     l = self._l.lowerLeaf(dm)
        #     l.set_name("l_pulled")
        #     new_vars = TorchVariables(position = dm._vars._position, velocity= dm._vars._velocity, parameter_variables=dm._vars._parameter_variables | self._vars._parameter_variables)
        #     return TorchLagrangian(l, var=new_vars)
        else:
            Jt = dm._J.transpose()
            l_subst = self._l.lowerLeaf(dm)
            H_subst=  l_subst.grad(dm._qdot) @ dm.qdot() - l_subst
            M_subst= self._S._M.lowerLeaf(dm)
            M_pulled = Jt @ M_subst @dm._J
            f_subst = self._S._f.lowerLeaf(dm)
            # f_pulled = Jt @ (M_subst @ dm._Jdotqdot + f_subst)
            f_pulled = Jt @ (f_subst)
            new_vars = TorchVariables(position = dm._vars._position, velocity= dm._vars._velocity, parameter_variables=dm._vars._parameter_variables | self._vars._parameter_variables)
            S_pulled = TorchSpec(M_pulled, f=f_pulled, var=new_vars)

            # return TorchLagrangian(l_subst, spec=S_pulled, hamiltonian=H_subst, var=new_vars)
            return TorchLagrangian(l_subst, spec=S_pulled,  var=new_vars)


        # new_state_variables = dm.state_variables()
        # new_parameters = {}
        # new_parameters.update(self._vars.parameters())
        # new_parameters.update(dm.params())
        # new_vars = Variables(state_variables=new_state_variables, parameters=new_parameters).toTorch()
        # if hasattr(dm, '_refTraj'):
        #     refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        # else:
        #     refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        # J_ref = dm._J
        # if self.is_dynamic():
        #     return TorchLagrangian(l, var=new_vars, J_ref=J_ref, ref_names=self.ref_names())
        # else:
            # return TorchLagrangian(l, var=new_vars, ref_names=self.ref_names())

    def __add__(self, b):
        assert isinstance(b, TorchLagrangian)
        # checkCompatability(self, b)
        # refTrajs = joinRefTrajs(self._refTrajs, b._refTrajs)
        # ref_names = []
        # if self.is_dynamic():
        #     ref_names += self.ref_names()
        #     J_ref = self._J_ref
        # if b.is_dynamic():
        #     ref_names += b.ref_names()
        #     J_ref = b._J_ref
        # if len(ref_names) > 0:
        #     ref_arguments = {'ref_names': ref_names, 'J_ref': J_ref}
        # else:
        #     ref_arguments = {}
        new_vars = self._vars + b._vars
        # return  TorchLagrangian(self._l + b._l, spec=self._S + b._S, hamiltonian=self._H + b._H, var=new_vars)
        return  TorchLagrangian(self._l, spec=self._S + b._S, var=new_vars)
    
class Lagrangian(object):
    """description"""

    def __init__(self, l: ca.SX, **kwargs):
        assert isinstance(l, ca.SX)
        self._l = l
        self.process_arguments(**kwargs)

    def process_arguments(self, **kwargs):
        self._x_ref_name = "x_ref"
        self._xdot_ref_name = "xdot_ref"
        self._xddot_ref_name = "xddot_ref"
        if 'x' in kwargs:
            self._vars = Variables(state_variables={"x": kwargs.get('x'), "xdot": kwargs.get('xdot')})
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
        self._rel = False
        self._refTrajs = []
        if 'ref_names' in kwargs:
            ref_names = kwargs.get('ref_names')
            self._x_ref_name = ref_names[0]
            self._xdot_ref_name = ref_names[1]
            self._xddot_ref_name = ref_names[2]
        if 'refTrajs' in kwargs:
            self._refTrajs = kwargs.get('refTrajs')
            self._rel = len(self._refTrajs) > 0
        if self.is_dynamic():
            self._J_ref_inv = np.identity(self.x_ref().size()[0])
        if "J_ref" in kwargs:
            self._J_ref = kwargs.get("J_ref")
            logging.info("Casadi pseudo inverse is used in Lagrangian")
            self._J_ref_inv = ca.mtimes(ca.transpose(self._J_ref), ca.inv(ca.mtimes(self._J_ref, ca.transpose(self._J_ref)) + np.identity(self.x_ref().size()[0]) * eps))
        if not self.is_dynamic() and 'spec' in kwargs and 'hamiltonian' in kwargs:
            self._H = kwargs.get('hamiltonian')
            self._S = kwargs.get('spec')
        else:
            # print("EulerLagrange Applied with ", self._l, kwargs)
            self.applyEulerLagrange()

    def x_ref(self):
        return self._vars.parameter_by_name(self._x_ref_name)

    def xdot_ref(self):
        return self._vars.parameter_by_name(self._xdot_ref_name)

    def x(self):
        return self._vars.position_variable()

    def xdot(self):
        return self._vars.velocity_variable()

    def xdot_rel(self, ref_sign: int = 1):
        if self.is_dynamic():
            return self.xdot() - ca.mtimes(self._J_ref_inv, self.xdot_ref()) * ref_sign
        else:
            return self.xdot()

    def __add__(self, b):
        assert isinstance(b, Lagrangian)
        checkCompatability(self, b)
        refTrajs = joinRefTrajs(self._refTrajs, b._refTrajs)
        ref_names = []
        if self.is_dynamic():
            ref_names += self.ref_names()
            J_ref = self._J_ref
        if b.is_dynamic():
            ref_names += b.ref_names()
            J_ref = b._J_ref
        if len(ref_names) > 0:
            ref_arguments = {'ref_names': ref_names, 'J_ref': J_ref}
        else:
            ref_arguments = {}
        new_vars = self._vars + b._vars
        return Lagrangian(self._l + b._l, spec=self._S + b._S, hamiltonian=self._H + b._H, var=new_vars, **ref_arguments)

    def is_dynamic(self) -> bool:
        logging.debug(f"Lagrangian is dynamic: {self._x_ref_name in self._vars.parameters()}")
        return self._x_ref_name in self._vars.parameters()


    def applyEulerLagrange(self):
        dL_dxdot = ca.gradient(self._l, self.xdot())
        dL_dx = ca.gradient(self._l, self.x())
        d2L_dxdxdot = ca.jacobian(dL_dx, self.xdot())
        d2L_dxdot2 = ca.jacobian(dL_dxdot, self.xdot())
        f_rel = np.zeros(self.x().size()[0])
        en_rel = np.zeros(1)

        if self.is_dynamic():
            x_ref = self._vars.parameters()[self._x_ref_name]
            xdot_ref = self._vars.parameters()[self._xdot_ref_name]
            xddot_ref = self._vars.parameters()[self._xddot_ref_name]
            dL_dxpdot = ca.gradient(self._l, xdot_ref)
            d2L_dxdotdxpdot = ca.jacobian(dL_dxdot, xdot_ref)
            d2L_dxdotdxp = ca.jacobian(dL_dxdot, x_ref)
            f_rel1 = ca.mtimes(d2L_dxdotdxpdot, xddot_ref)
            f_rel2 = ca.mtimes(d2L_dxdotdxp, xdot_ref)
            f_rel += f_rel1 + f_rel2
            en_rel += ca.dot(dL_dxpdot, xdot_ref)

        F = d2L_dxdxdot
        f_e = -dL_dx
        M = d2L_dxdot2
        f = ca.mtimes(ca.transpose(F), self.xdot()) + f_e + f_rel
        self._H = ca.dot(dL_dxdot, self.xdot()) - self._l + en_rel
        self._S = Spec(M, f=f, var=self._vars, refTrajs=self._refTrajs, l=self._l)

    def concretize(self):
        self._S.concretize()
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        self._funs = CasadiFunctionWrapper(
            "funs", var, {"H": self._H}
        )
        self._L_funs = CasadiFunctionWrapper(
            "L_funs", var, {"L": self._l}
        )

    def ref_names(self) -> list:
        return [self._x_ref_name, self._xdot_ref_name, self._xddot_ref_name]


    def evaluate(self, **kwargs):
        funs = self._funs.evaluate(**kwargs)
        H = funs['H']
        M, f, _ = self._S.evaluate(**kwargs)
        L = self._L_funs.evaluate(**kwargs)
        return M, f, H, L

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        l_subst = ca.substitute(self._l, self.x(), dm._phi)
        l_subst2 = ca.substitute(l_subst, self.xdot(), dm.phidot())
        new_state_variables = dm.state_variables()
        new_parameters = {}
        new_parameters.update(self._vars.parameters())
        new_parameters.update(dm.params())
        new_vars = Variables(state_variables=new_state_variables, parameters=new_parameters)
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        J_ref = dm._J
        if self.is_dynamic():
            return Lagrangian(l_subst2, var=new_vars, J_ref=J_ref, ref_names=self.ref_names())
        else:
            return Lagrangian(l_subst2, var=new_vars, ref_names=self.ref_names())

    def dynamic_pull(self, dm: DynamicDifferentialMap):
        l_pulled = self._l
        l_pulled_subst_x = ca.substitute(l_pulled, self.x(), dm._phi)
        l_pulled_subst_x_xdot = ca.substitute(l_pulled_subst_x, self.xdot(), dm.phidot())
        return Lagrangian(l_pulled_subst_x_xdot, var=dm._vars, ref_names=dm.ref_names())


class FinslerStructure(Lagrangian):
    def __init__(self, lg: ca.SX, **kwargs):
        self._lg = lg
        l = 0.5 * lg ** 2
        super().__init__(l, **kwargs)

    def concretize(self):
        super().concretize()
        self._funs_lg = CasadiFunctionWrapper(
            "funs", self._vars, {"Lg": self._lg}
        )

    def evaluate(self, **kwargs):
        M, f, l = super().evaluate(**kwargs)
        lg = self._funs_lg.evaluate(**kwargs)['Lg']
        return M, f, l, lg
