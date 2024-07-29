import casadi as ca
import numpy as np
from copy import deepcopy

from fabrics.diffGeometry.spec import Spec, checkCompatability, TorchSpec
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap, TorchDifferentialMap
from fabrics.diffGeometry.casadi_helpers import outerProduct

from fabrics.helpers.constants import eps
from fabrics.helpers.functions import joinRefTrajs
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper

import torch

# class EnergizedGeometry(Spec):
#     # Should not be used as it is not compliant with summation
#     # Only used for verification in testing
#     def __init__(self, g: Geometry, le: Lagrangian):
#         assert isinstance(le, Lagrangian)
#         assert isinstance(g, Geometry)
#         checkCompatability(le, g)
#         frac = outerProduct(g.xdot(), g.xdot()) / (
#             eps + ca.dot(g.xdot(), ca.mtimes(le._S._M, g.xdot()))
#         )
#         pe = np.identity(le.x().size()[0]) - ca.mtimes(le._S._M, frac)
#         f = le._S._f + ca.mtimes(pe, ca.mtimes(le._S._M, g._h) - le._S._f)
#         super().__init__(le._S._M, f=f, var=g._vars)
#         self._le = le

class TorchWeightedGeometry(TorchSpec):
    # Spec + Finler Lagrangian
    def __init__(self, **kwargs):
        le = kwargs.get("le")
        self._le = le
        if "g" in kwargs:
            g = kwargs.get("g")
            var = g._vars + le._vars
            if le._x != g._x:
                raise Exception("geometry and lagrangian with different space")
            self._le = le
            self._h = g._h
            self._M = le._S._M
            self._vars = var
            self._f = self._M @ self._h
        if "s" in kwargs:
            s = kwargs.get("s")
            if le._x != s._x:
                raise Exception("spec and lagrangian with different space")
            self._le = le
            if hasattr(s, '_f_subst') and hasattr(s, '_M_subst'):
                super().__init__(s._M, f=s._f, var=s._vars, M_subst=s._M_subst, f_subst=s._f_subst)
            else:
                super().__init__(s._M, f=s._f, var=s._vars)
        if "l_subst" in kwargs:
            self._l_subst = kwargs.get("l_subst")
        self._x = self._vars._position
        self._xdot = self._vars._velocity
        # frac = 1/(eps+ self.xdot().dot(self._le._S._M @ self.xdot()))
        frac = 1/(eps+ self.xdot().dot(self._M @ self.xdot()))
        
        self.frac = frac
        self._alpha = -frac * self.xdot().dot(self._f - self._le._S._f)
        self._alpha.set_name("alpha")
        self._xddot = -self._h
        self._xddot.set_name("xddot")

    def __add__(self, b):
        spec = super().__add__(b)
        le = self._le + b._le
        return TorchWeightedGeometry(s=spec, le=le) 
    
    def pull(self, dm: TorchDifferentialMap): 
        spec = super().pull(dm)
        le_pulled = self._le.pull(dm)
        l_subst = self._le._l.lowerLeaf(dm)
        return TorchWeightedGeometry(s=spec, le=le_pulled, l_subst = l_subst) 
    # def evaluate(self, x: torch.Tensor ,xdot: torch.Tensor):
    #     return [self._M(x,xdot), self._f(x,xdot), self._xddot(x,xdot), self._alpha(x,xdot)]
class WeightedGeometry(Spec):
    def __init__(self, **kwargs):
        self._x_ref_name = "x_ref"
        self._xdot_ref_name = "xdot_ref"
        self._xddot_ref_name = "xddot_ref"
        le = kwargs.get("le")
        assert isinstance(le, Lagrangian)
        if 'ref_names' in kwargs:
            ref_names = kwargs.get('ref_names')
            self._x_ref_name = ref_names[0]
            self._xdot_ref_name = ref_names[1]
            self._xddot_ref_name = ref_names[2]
        if "g" in kwargs:
            g = kwargs.get("g")
            checkCompatability(le, g)
            var = g._vars + le._vars
            self._refTrajs = joinRefTrajs(le._refTrajs, g._refTrajs)
            self._le = le
            self._h = g._h
            self._M = le._S.M()
            self._vars = var
        if "s" in kwargs:
            s = kwargs.get("s")
            checkCompatability(le, s)
            self._le = le
            refTrajs = joinRefTrajs(le._refTrajs, s._refTrajs)
            super().__init__(s.M(), f=s.f(), var=s._vars, refTrajs=refTrajs, ref_names=self.ref_names())
        if "M_subst" in kwargs:
            self._M_subst = kwargs.get("M_subst")
        if "f_subst" in kwargs:
            self._f_subst = kwargs.get("f_subst")
        if "l_subst" in kwargs:
            self._l_subst = kwargs.get("l_subst")

    def __add__(self, b):
        spec = super().__add__(b)
        le = self._le + b._le
        return WeightedGeometry(s=spec, le=le, ref_names=spec.ref_names())

    def computeAlpha(self, ref_sign: int = 1): #proposition 4.22 in Optimization Fabrics
        xdot = self._le.xdot_rel(ref_sign=ref_sign)
        frac = 1 / (
            eps + ca.dot(xdot, ca.mtimes(self._le._S.M(), xdot))
        )
        self.frac = frac
        self._alpha = -frac * ca.dot(xdot, self.f() - self._le._S.f())
        self._xddot = -self.h()

    def concretize(self, ref_sign: int = 1):
        self.computeAlpha(ref_sign=ref_sign)
        self._xddot = -self.h()
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        
        """
        self._funs = ca.Function(
            "M", var, [self.M(), self.f(), self._xddot, self._alpha]
        )
        """
        
        self._funs = CasadiFunctionWrapper(
                "funs", var, {"M": self.M(), 'Minv': self.Minv(), 'f': self.f(), 'xddot': self._xddot, 'alpha': self._alpha, 'frac': self.frac,'lag_M':self._le._S._M, 'lag_f': self._le._S.f()}
        )
        
    def evaluate(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        M = evaluations['M']
        M_cond = np.linalg.cond(M)
        U,S,V = np.linalg.svd(M)
        Minv = evaluations['Minv']
        f = evaluations['f']
        xddot = evaluations['xddot']
        alpha = evaluations['alpha']
        frac = evaluations['frac']
        lag_f = evaluations['lag_f']
        lag_M = evaluations['lag_M']
        return [M, Minv, f, xddot, alpha, frac, lag_M, lag_f]

    def pull(self, dm: DifferentialMap):
        spec = super().pull(dm)
        le_pulled = self._le.pull(dm)
        l_subst = ca.substitute(self._le._l, self.x(), dm._phi)
        l_subst = ca.substitute(l_subst, self.xdot(), dm.phidot())
        return WeightedGeometry(s=spec, le=le_pulled, ref_names=self.ref_names(), M_subst=spec._M_subst, f_subst=spec._f_subst, l_subst= l_subst)

    def dynamic_pull(self, dm: DynamicDifferentialMap):
        spec = super().dynamic_pull(dm)
        le_pulled = self._le.dynamic_pull(dm)
        return WeightedGeometry(s=spec, le=le_pulled, ref_names=dm.ref_names())

    def x(self):
        return self._le.x()

    def xdot(self):
        return self._le.xdot()
