import logging
from copy import deepcopy
from typing import Dict, List, Optional

import casadi as ca
import deprecation
import numpy as np

from fabrics.diffGeometry.spec import TorchSpec
from forwardkinematics.fksCommon.fk import ForwardKinematics
from forwardkinematics.urdfFks.urdfFk import LinkNotInURDFError
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.goals.sub_goal import SubGoal
from pyquaternion import Quaternion

from fabrics import __version__
from fabrics.components.energies.execution_energies import ExecutionLagrangian, TorchExecutionLagrangian
from fabrics.components.leaves.attractor import GenericAttractor, TorchGenericAttractor
from fabrics.components.leaves.dynamic_attractor import GenericDynamicAttractor
from fabrics.components.leaves.dynamic_geometry import (
    DynamicObstacleLeaf, GenericDynamicGeometryLeaf)
from fabrics.components.leaves.geometry import (AvoidanceLeaf,
                                                CapsuleCuboidLeaf,
                                                CapsuleSphereLeaf,
                                                ESDFGeometryLeaf,
                                                GenericGeometryLeaf, LimitLeaf, TorchGenericGeometryLeaf, TorchLimitLeaf,
                                                # TorchGenericGeometryLeaf,
                                                ObstacleLeaf,
                                                PlaneConstraintGeometryLeaf,
                                                SelfCollisionLeaf,
                                                SphereCuboidLeaf)
from fabrics.components.leaves.leaf import Leaf
from fabrics.diffGeometry.diffMap import (DifferentialMap, TorchDifferentialMap,
                                          DynamicDifferentialMap)
from fabrics.diffGeometry.energized_geometry import WeightedGeometry, TorchWeightedGeometry
from fabrics.diffGeometry.energy import Lagrangian, TorchLagrangian
from fabrics.diffGeometry.geometry import Geometry, TorchGeometry
from fabrics.diffGeometry.speedControl import Damper, TorchDamper
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper, TorchFunctionWrapper
from fabrics.helpers.constants import eps
from fabrics.helpers.exceptions import ExpressionSparseError
from fabrics.helpers.functions import is_sparse, parse_symbolic_input
from fabrics.helpers.geometric_primitives import Sphere
from fabrics.helpers.variables import Variables, TorchVariables
from fabrics.planner.configuration_classes import (FabricPlannerConfig, TorchConfig,
                                                   ProblemConfiguration)

import torch,functorch

class InvalidRotationAnglesError(Exception):
    pass

class LeafNotFoundError(Exception):
    pass


@deprecation.deprecated(deprecated_in="0.8.8", removed_in="0.9",
                        current_version=__version__,
                        details="Remove the goal attribute angle and rotate the position before passing it into compute_action.")
def compute_rotation_matrix(angles) -> np.ndarray:
    if isinstance(angles, float):
        angle = angles
        return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    elif isinstance(angles, list) and len(angles) == 4:
        quaternion = Quaternion(angles)
        return quaternion.rotation_matrix
    elif isinstance(angles, ca.SX):
        return angles
    else:
        raise(InvalidRotationAnglesError)

class TorchPlanner(object):
    _dof : int
    _config: TorchConfig
    leavs: Dict[str, Leaf]
    _ref_sign: int

    def __init__(self, dof:int, forward_kinematics:ForwardKinematics, **kwargs):
        self._dof = dof
        self._config = TorchConfig(**kwargs)
        self._forward_kinematics = forward_kinematics
        self.initialize_joint_varibles()
        self.set_base_geometry()
        self._target_velocity = torch.zeros(self._dof,dtype=torch.float64) #cart dof
        self._ref_sign = 1
        self.leaves = {}


    
    """ INIT"""

    def initialize_joint_varibles(self):
        q = ca.SX.sym("q", self._dof)
        qdot = ca.SX.sym("qdot", self._dof)
        self._variables = Variables(state_variables={"q": q, "qdot": qdot})


    def set_base_geometry(self):
        q = self._variables.position_variable()
        qdot = self._variables.velocity_variable() 
        base_energy = TorchFunctionWrapper(expression=self._config.base_energy,variables=self._variables.toTorch(), ex_input=["q","qdot"], name="base_energy")
        base_M = TorchFunctionWrapper(expression=self._config.base_M,variables=self._variables.toTorch(), ex_input=["q","qdot"], name="base_energy")
        self.base_M = base_M
        base_f = TorchFunctionWrapper(expression=self._config.base_f,variables=self._variables.toTorch(), ex_input=["q","qdot"], name="base_energy")
        base_spec = TorchSpec(base_M, f=base_f, var=self._variables.toTorch())
        base_h = TorchFunctionWrapper(expression=lambda : torch.zeros(self._dof,dtype=torch.float64), variables=self._variables.toTorch(), ex_input=[])
        base_geometry = TorchGeometry(h = base_h, var=self._variables.toTorch())
        base_lagrangian = TorchLagrangian(base_energy, spec=base_spec, var=self._variables.toTorch())
        self._geometry = TorchWeightedGeometry(g=base_geometry, le=base_lagrangian)
        self.base_energy = base_energy
    
    """ ADDING COMPONENTS"""
    def add_geometry(
        self, forward_map: TorchDifferentialMap, lagrangian: TorchLagrangian, geometry: Geometry , isLimit: bool = False
    ) -> None:
        weighted_geometry =TorchWeightedGeometry(g=geometry, le=lagrangian, isLimit = isLimit)
        self.add_weighted_geometry(forward_map, weighted_geometry)
    
    def add_weighted_geometry(
        self, forward_map: TorchDifferentialMap, weighted_geometry: TorchWeightedGeometry
    ) -> None:
        pulled_geometry = weighted_geometry.pull(forward_map)

        self._geometry += pulled_geometry
        # self._variables = self._variables + pulled_geometry._vars #이것도 마찬가지로 문제가 될 수 있는 부분
    
    def add_forcing_geometry(
        self,
        forward_map: TorchDifferentialMap,
        lagrangian: TorchLagrangian,
        geometry: TorchGeometry,
        prime_forcing_leaf: bool,
    ) -> None:
        if not hasattr(self, '_forced_geometry'):
            self._forced_geometry = deepcopy(self._geometry)
        self._pulled_attractor = TorchWeightedGeometry(g=geometry,le=lagrangian).pull(forward_map)
        self._forced_geometry += TorchWeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(forward_map)
        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map
        # self._variables = self._variables + self._forced_geometry._vars ### 일단 지금의 경우에는 같아서 무시하지만 나중에 문제가 될 수 있는 부분임
    
    def add_limit_geometry(
            self,
            limits: torch.Tensor,
            ) -> None:
        lower_limit_geometry = TorchLimitLeaf(self._variables.toTorch(), limits[:,0], 0)
        lower_limit_geometry.set_geometry(self._config.limit_geometry)
        # lower_limit_geometry.set_finsler_structure(self._config.limit_finsler)
        lower_limit_geometry.set_lag(self._config.limit_finsler,self._config.limit_M, self._config.limit_f)
        upper_limit_geometry = TorchLimitLeaf(self._variables.toTorch(), limits[:,1], 1)
        upper_limit_geometry.set_geometry(self._config.limit_geometry)
        # upper_limit_geometry.set_finsler_structure(self._config.limit_finsler)
        upper_limit_geometry.set_lag(self._config.limit_finsler,self._config.limit_M, self._config.limit_f)
        self.add_leaf(lower_limit_geometry)
        self.add_leaf(upper_limit_geometry)

    def add_leaf(self, leaf: Leaf, prime_leaf: bool= False) -> None:
        if isinstance(leaf, TorchGenericAttractor):
            self.add_forcing_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry(), prime_leaf)
        elif isinstance(leaf, TorchLimitLeaf):
            self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry(), isLimit=True)
        elif isinstance(leaf, TorchGenericGeometryLeaf):
            self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())
        self.leaves[leaf._leaf_name] = leaf
    
    def get_leaves(self, leaf_names:list) -> List[Leaf]:
        leaves = []
        for leaf_name in leaf_names:
            if leaf_name not in self.leaves:
                error_message = f"Leaf with name {leaf_name} not in leaves.\n"
                error_message = f"Possible leaves are {list(self.leaves.keys())}."
                raise LeafNotFoundError(error_message)
            leaves.append(self.leaves[leaf_name])
        return leaves
    
    def get_differential_map(self, sub_goal_index: int, sub_goal: SubGoal):
        if sub_goal.type() == 'staticJointSpaceSubGoal':
            return self._variables.position_variable()[sub_goal.indices()]
        else:
            fk_child = self.get_forward_kinematics(sub_goal.child_link())
            try:
                fk_parent = self.get_forward_kinematics(sub_goal.parent_link())
            except LinkNotInURDFError as e:
                fk_parent = ca.SX(np.zeros(3))
            angles = sub_goal.angle()

            if angles and isinstance(angles, list) and len(angles) == 4:
                logging.warning(
                    "Subgoal attribute 'angle' deprecated. " \
                    +"Remove the goal attribute angle and rotate the" \
                    +"position before passing it into"\
                    +"compute_action."\
                    +"it's here"
                )
                angles = ca.SX.sym(f"angle_goal_{sub_goal_index}", 3, 3)
                self._variables.add_parameter(f'angle_goal_{sub_goal_index}', angles)
                # rotation
                R = compute_rotation_matrix(angles)
                fk_child = ca.mtimes(R, fk_child)
                fk_parent = ca.mtimes(R, fk_parent)
            elif angles:
                logging.warning(
                    "Subgoal attribute 'angle' deprecated. " \
                    +"Remove the goal attribute angle and rotate the" \
                    +"position before passing it into"\
                    +"compute_action."
                )
                R = compute_rotation_matrix(angles)
                fk_child = ca.mtimes(R, fk_child)
                fk_parent = ca.mtimes(R, fk_parent)
            return fk_child[sub_goal.indices()] - fk_parent[sub_goal.indices()]

    def get_forward_kinematics(self, link_name, position_only: bool = True) -> ca.SX:
        if isinstance(link_name, ca.SX):
            return link_name
        fk = self._forward_kinematics.casadi(
                self._variables.position_variable(),
                link_name,
                position_only=position_only
            )
        return fk
    
    def set_goal_component(self, goal: GoalComposition):
        # Adds default attractor
        for j, sub_goal in enumerate(goal.sub_goals()):
            fk_sub_goal = self.get_differential_map(j, sub_goal)
            if is_sparse(fk_sub_goal):
                raise ExpressionSparseError()
            self._variables.add_parameter(f'x_goal_{j}', ca.SX.sym(f'x_goal_{j}', sub_goal.dimension()))
            attractor = TorchGenericAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            attractor.set_geom(self._config.attractor_geom)
            attractor.set_lag(self._config.attractor_metric, self._config.attractor_force)
            self.add_leaf(attractor, prime_leaf=sub_goal.is_primary_goal())

    # def test_init_function(self, **kwargs):
    #     # print("base energy : ", self._geometry._le._l(**kwargs))
    #     print("q", kwargs['q'])
    #     print("qdot", kwargs['qdot'])
    #     print("leavs", self.get_leaves(["goal_0_leaf"]))
    #     # attractor = self.leaves['goal_0_leaf']
    #     # pulled_geom = TorchWeightedGeometry(
    #     #     g=attractor.geometry(), le=attractor.lagrangian()
    #     # ).pull(attractor.map())
    #     # J = attractor._map._J
    #     # M_subst = attractor._lag._S._M.lowerLeaf(attractor.map())
    #     # pulled_M = J.transpose() @ M_subst @ J
    #     # print(" J", J(**kwargs))
    #     # print("energy ", self._geometry._le._l(**kwargs))
    #     # print("grad test", self._geometry._le._dL_dxdot(**kwargs))
    #     # print("attractor psi", attractor.psi.lowerLeaf(attractor.map())(**kwargs))
    #     # print("attractor h", attractor.h_psi.lowerLeaf(attractor.map())(**kwargs))
    #     # print("pulled M",  self._forced_geometry._M(**kwargs))
    #     # print("pulled f", self._geometry._f(**kwargs))

    #     print("xddot", self._xddot(**kwargs))

        # print("q", kwargs['q'])
        # # print("J", attractor._map._J(**kwargs))

    def set_execution_energy(self, execution_lagrangian: Lagrangian):
        composed_geometry = TorchGeometry(s=self._geometry)
        self._execution_lagrangian = execution_lagrangian
        self._execution_geometry = TorchWeightedGeometry(
            g=composed_geometry, le= execution_lagrangian
        )

        forced_geometry = TorchGeometry(s=self._forced_geometry)
        self._forced_speed_controlled_geometry = TorchWeightedGeometry(
                g=forced_geometry, le=execution_lagrangian
            )
    
    def set_speed_control(self):
        x_psi = self._forced_variables.position_variable()
        dm_psi = self._forced_forward_map
        exLag = self._execution_lagrangian
        beta_expression = self._config.damper_beta
        eta_expression = self._config.damper_eta
        self._damper = TorchDamper(beta_expression, eta_expression, x_psi, dm_psi, exLag._l)
        if self._config.forcing_type in ['speed-controlled']:
            eta = self._damper.substitute_eta()
            self.eta = eta
            eta.set_name("eta")
            a_ex = (
                eta * self._execution_geometry._alpha
                + (1 - eta) * self._forced_speed_controlled_geometry._alpha
            )
            self.a_ex =a_ex
            self.a_ex.set_name("a_ex")
            beta_subst = self._damper.substitute_beta(-a_ex, -self._geometry._alpha)
            beta_subst.set_name("beta")
            self.beta_subst = beta_subst
            self._xddot = self._forced_geometry._xddot - (a_ex + beta_subst) * (
                self._geometry.xdot()
                - self._forced_geometry._Minv @ self._target_velocity
            )
            self._xddot.set_name("xddot") 
            self._forced_geometry._xddot.set_name("geom xddot")

            #xddot = self._forced_geometry._xddot

    def set_components(
        self,
        collision_links: Optional[list] = None,
        self_collision_pairs: Optional[dict] = None,
        collision_links_esdf: Optional[list] = None,
        goal: Optional[GoalComposition] = None,
        limits: Optional[torch.Tensor] = None,
        number_obstacles: int = 1,
        number_dynamic_obstacles: int = 0,
        number_obstacles_cuboid: int = 0,
        number_plane_constraints: int = 0,
        dynamic_obstacle_dimension: int = 3,
    ):
        if limits is not None:
            # for joint_index in range(len(limits)):
            self.add_limit_geometry(limits)
        if goal:
            self.set_goal_component(goal)
            # Adds default execution energy
            execution_energy = TorchExecutionLagrangian(self._variables.toTorch())
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()

            #xddot = self._forced_geometry._xddot
            

    def compute_action(self, **kwargs):
        """
        Computes action based on the states passed.

        The variables passed are the joint states, and the goal position.
        The action is nullified if its magnitude is very large or very small.
        """
        TorchFunctionWrapper.reset_all_caches()

        # evaluations = self.xddot().evaluate(**kwargs) ##여기서는action 값 밖에 안나옴
        # action = evaluations["action"]
        # Debugging
        #logging.debug(f"a_ex: {evaluations['a_ex']}")
        #logging.debug(f"alhpa_forced_geometry: {evaluations['alpha_forced_geometry']}")
        #logging.debug(f"alpha_geometry: {evaluations['alpha_geometry']}")
        #logging.debug(f"beta : {evaluations['beta']}")
        # self._execution_geometry._alpha(**kwargs)
        # self._geometry._le._S._M(**kwargs)
        # a_ex = self.a_ex(**kwargs)
        # beta = self.beta_subst(**kwargs)
        # print("eta2", self.eta(**kwargs))
        # print("a_ex2", self.a_ex(**kwargs))
        # print("beta2", self.beta_subst(**kwargs))
        # print("alpha2", self._forced_speed_controlled_geometry._alpha(**kwargs))
        # print("base alpha2", self._geometry._alpha(**kwargs))

        # # self._forced_geometry._xddot(**kwargs)
        # # self._forced_geometry._Minv(**kwargs)

        # attractor_frac =  self._forced_geometry.frac
        # attractor_xddot =  self._forced_geometry._xddot
        # attractor_alpha =  self._forced_geometry._alpha
        # attractor_f =  self._forced_geometry._f
        # le_M = self._forced_geometry._le._S._M
        # attractor_lag_f =  self._forced_geometry._le._S._f
        # M = self._forced_geometry._M(**kwargs)
        # f = self._forced_geometry._f(**kwargs)
        # xddot = self._forced_geometry._xddot(**kwargs)
        # # alpha = self._forced_geometry._alpha(**kwargs)
        # frac = self._forced_geometry.frac(**kwargs)
        # le_M = self._forced_geometry._le._S._M(**kwargs)
        # le_f = self._forced_geometry._le._S._f(**kwargs)
        # limit = self.leaves['limit_joint_1_leaf']
        # limitgeom = TorchWeightedGeometry(g=limit._geo, le=limit._lag, isLimit = True).pull(limit.map())
        # print("M2:", limitgeom._M(**kwargs))
        # print(" f2:", limitgeom._f(**kwargs))
        # print(" xddot2:", limitgeom._xddot(**kwargs))
        # print("frac2:", limitgeom.frac(**kwargs))
        # print("alpha2:", limitgeom._le._S._f(**kwargs))

        # print("le_M:", limmile_M)
        # print("le_f:", le_f)
        # print("M2", attractor_M(**kwargs))
        # print("f2", attractor_f(**kwargs))
        # print("xddot2", attractor_xddot(**kwargs))
        # print("frac2", attractor_frac(**kwargs))
        # # print("le_M2", attractor_le_M(**kwargs))
        # # print("le_f2", attractor_le_f(**kwargs))
        # print("alpha2", attractor_alpha(**kwargs))
        # print("M2")
        # print("forced xddot", self._forced_geometry._xddot(**kwargs))
        # q= self._variables._state_variable_names[0]
        # qdot= self._variables._state_variable_names[1]
        # x = attractor._map._phi(**kwargs)
        # xdot = attractor._map._J(**kwargs) @ kwargs[qdot]
        # metric = self._config.attractor_metric
        # def lag(q,qdot):
        #     x = attractor._map._phi(**kwargs)
        #     xdot = attractor._map._J(**kwargs) @ kwargs[qdot]
        #     return torch.sum(xdot *( metric(x,xdot) @ xdot),dim=-1)
        # l = TorchFunctionWrapper(expression=lag, variables=self._geometry._vars, ex_input=[q,qdot])
        # print("metric", metric(x,xdot))
        # print("lag psi2", torch.sum(xdot* (metric(x,xdot).type(torch.float64)@xdot), dim=-1))
        # M = l.hessian(qdot,qdot)
        # F = l.hessian(q, qdot)
        # f_e = -l.grad(q)
        # dL_dxdot = l.grad(qdot)
        # dL_dx = l.grad(q)
        # d2L_dxdxdot = dL_dx.grad(qdot, end_grad=True)
        # # d2L_dxdot2 = dL_dxdot.grad(qdot, end_grad=True)
        # M(**kwargs)
        # F(**kwargs)
        # f_e = -dL_dx
        # M = d2L_dxdot2
        # f = F.transpose() @ kwargs[qdot] + f_e
        # f_value = f(**kwargs)
        # frac = pulled_geom.frac(**kwargs)
        # alpha = -frac * kwargs[qdot]*(pulled_geom._f(**kwargs) - f_value)
        # attractor = self.leaves['goal_0_leaf']
        # psi = attractor.psi
        
        # h = psi.grad(attractor._xdot)
        # print("psi", attractor.psi.lowerLeaf(attractor.map())(**kwargs))
        # print("psi input", psi._ex_input)
        # print("xdot", attractor._xdot)
        # print("h", h.lowerLeaf(attractor.map())(**kwargs))
        # not_pulled_geom = TorchWeightedGeometry(g=attractor.geometry(), le=attractor.lagrangian())
        # pulled_geom = not_pulled_geom.pull(attractor.map())
        # print("M2", pulled_geom._M(**kwargs))
        # print("f2", pulled_geom._f(**kwargs))
        # print("h2", pulled_geom._h(**kwargs))
        # print("xddot2", pulled_geom._xddot(**kwargs))
        # print("frac2", pulled_geom.frac(**kwargs))
        # print("l2", pulled_geom._le._l(**kwargs))
        # dL_dxdot.set_name("dL_dxdot")
        # dL_dx = l.grad("q")
        # dL_dx.set_name("dL_dx")
        # d2L_dxdxdot = dL_dx.grad("qdot")
        # d2L_dxdot2 = dL_dxdot.grad("qdot")
        # F = d2L_dxdxdot
        # F.set_name("F:d2L_dxdxdot")
        # f_e = -dL_dx
        # M = d2L_dxdot2
        # M.set_name("M:d2L_dxdot2")
        # f = F.transpose() @ kwargs["qdot"] + f_e
        # f.set_name("lagrange f")

        # print("le_M2", pulled_geom._le._S._M(**kwargs))
        # print("le_f2", pulled_geom._le._S._f(**kwargs))
        # print("alpha2", pulled_geom._alpha(**kwargs))


        # print("M2", not_pulled_geom._M.lowerLeaf(attractor.map())(**kwargs))
        # print("f2", not_pulled_geom._f.lowerLeaf(attractor.map())(**kwargs))
        # print("h2", not_pulled_geom._h.lowerLeaf(attractor.map())(**kwargs))
        # print("xddot2", not_pulled_geom._xddot.lowerLeaf(attractor.map())(**kwargs))
        # print("frac2", not_pulled_geom.frac.lowerLeaf(attractor.map())(**kwargs))
        # print("l2", not_pulled_geom._le._l.lowerLeaf(attractor.map())(**kwargs))
        # print("le_M2", not_pulled_geom._le._S._M.lowerLeaf(attractor.map())(**kwargs))
        # print("le_f2", not_pulled_geom._le._S._f.lowerLeaf(attractor.map())(**kwargs))
        # print("alpha2", not_pulled_geom._alpha.lowerLeaf(attractor.map())(**kwargs))
        # print("baseM2", self._geometry._M(**kwargs))
        # print("basef2", self._geometry._f(**kwargs))
        # print("basexddot2", self._geometry._xddot(**kwargs))
        # print("basefrac2", self._geometry.frac(**kwargs))
        # print("basealpha2", self._geometry._alpha(**kwargs))
        # print("basele_M", self._geometry._le._S._M(**kwargs))
        # print("basele_f", self._geometry._le._S._f(**kwargs))
        # print("force M2", self._forced_geometry._M(**kwargs))
        # print("force f2", self._forced_geometry._f(**kwargs))
        # print("force xddot2", self._forced_geometry._xddot(**kwargs))
        # print("force frac2", self._forced_geometry.frac(**kwargs))
        # print("force le_l", self._forced_geometry._le._l(**kwargs))
        # print("force le_M", self._forced_geometry._le._S._M(**kwargs))
        # print("force le_f", self._forced_geometry._le._S._f(**kwargs))
        # print("exe M2", self._execution_geometry._M(**kwargs))
        # print("exe f2", self._execution_geometry._f(**kwargs))
        # print("exe xddot2", self._execution_geometry._xddot(**kwargs))
        # print("exe frac2", self._execution_geometry.frac(**kwargs))
        # print("exe alpha2", self._execution_geometry._alpha(**kwargs))
        # print("exe le_M", self._execution_geometry._le._S._M(**kwargs))
        # print("exe le_f", self._execution_geometry._le._S._f(**kwargs))
        # limit = self.leaves[f"limit_{0}_leaf"]
        # limit_geom = TorchWeightedGeometry(g= limit._geo, le = limit._lag, isLimit=True).pull(limit.map())
        # print("liit M2", limit_geom._M(**kwargs))
        # print("limit f2", limit_geom._f(**kwargs))
        # print("limit xddot2", limit_geom._xddot(**kwargs))
        # print("limit frac2", limit_geom.frac(**kwargs))
        # print("limit alpha2", limit_geom._alpha(**kwargs))
        # print("limit le_l2", limit_geom._le._l(**kwargs))
        # print("limit le_M2", limit_geom._le._S._M(**kwargs))
        # print("limit le_f2", limit_geom._le._S._f(**kwargs))
        # print("a_ex", self.a_ex(**kwargs))
        # print("alpha", self._geometry._alpha(**kwargs))
        # print("forced_minv", self._forced_geometry._Minv(**kwargs))
        # forced_xddot(**kwargs)
        
        # h = attractor._geo._h.lowerLeaf(attractor.map())
        # print("attractor geom h2", h(**kwargs))
        # phi = attractor._map._phi(**kwargs)
        # Jdot = attractor._map._Jdot(**kwargs)
        # M_subst = self._forced_geometry._M_subst(**kwargs)
        # f_subst = self._forced_geometry._f_subst(**kwargs)
        # print("M_subst2", self._pulled_attractor._M_subst(**kwargs))
        # print("f_subst2", self._pulled_attractor._f_subst(**kwargs))
        # print("l_subst2", self._pulled_attractor._l_subst(**kwargs))
        # M_subst = attractor._lag._S._M.lowerLeaf(attractor.map())
        # # print("M_subst", J(**kwargs))
        # geom = self._forced_speed_controlled_geometry
        # exe_alpha = self._execution_geometry._alpha(**kwargs)
        # force_alpha = self._forced_speed_controlled_geometry._alpha(**kwargs)
        # force_f = self._forced_speed_controlled_geometry._f(**kwargs)
        # M = geom._M(**kwargs)
        # U,S,V = torch.svd(M)
        # f = geom._f(**kwargs)
        # alpha = geom._alpha(**kwargs)
        # frac = geom.frac(**kwargs)
        # lag_f = geom._le._S._f(**kwargs)
        # print("M2", M)
        # print("S2", S)
        # print("f", f)
        # print("alpha2", alpha)
        # print("frac2", frac)
        # print("lag_f2", lag_f)
        # print("Phi2", phi)
        # print("J2",J)
        # print("Jdot2", Jdot)
        # force_alpha = self._forced_geometry._alpha(**kwargs)
        # print("leves",self.get_leaves())
        # l = TorchFunctionWrapper(expression=self._config.attractor_l, variables=self._variables.toTorch(), ex_input = ["q","qdot"])
        # dldx_sym = TorchFunctionWrapper(expression=self._config.attractor_dldx, variables=self._variables.toTorch(), ex_input=["q","qdot"])
        # dldxdxdot_sym = TorchFunctionWrapper(expression=self._config.attractor_dldxdxdot, variables=self._variables.toTorch(), ex_input=["q","qdot"])
        # f_sym = TorchFunctionWrapper(expression=self._config.attractor_force, variables=self._variables.toTorch(), ex_input=["q","qdot"])
        # psi =  TorchFunctionWrapper(expression=self._config.attractor_potential, variables=self._variables.toTorch(), ex_input=["q","qdot"])
        # h_sym = TorchFunctionWrapper(expression=self._config.attractor_geom, variables=self._variables.toTorch(), ex_input=["q","qdot"])
        # h= psi.grad("q")
        # print("h", h(**kwargs))
        # print("h_sym", h_sym(**kwargs))
        # dldx = l.grad("q")
        # dldxdxdot = dldx.grad("qdot")
        # f = dldxdxdot.transpose()@self._geometry.xdot() -dldx
        # self.base_M(**kwargs)

        # print("exinput", len([kwargs[input] for input in l._ex_input]))
        # print("l", l(**kwargs))
        # print("dldx", dldx(**kwargs))
        # print("dLdx sym", dldx_sym(**kwargs))
        # print("dldxdxdot", dldxdxdot(**kwargs))
        # print("dLdxdxdot sym", dldxdxdot_sym(**kwargs))
        # print("f", f(**kwargs))
        # print("f", f_sym(**kwargs))
        action = self._xddot(**kwargs)
        print("grad_count", TorchFunctionWrapper.grad_count)
        TorchFunctionWrapper.grad_count = 0
        # ex = self._config.attractor_metric
        # f = TorchFunctionWrapper(expression=ex, variables=self._geometry._vars, ex_input=self._geometry._vars.position_velocity_variables())
        # f_grad = f.grad("q")
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)
        # f_grad._func(**kwargs)

        # action = torch.zeros(self._dof)
        # action_magnitude = torch.linalg.norm(action, dim=-1)

        # if action_magnitude < eps:
        #     # logging.warning(f"Fabrics: Avoiding small action with magnitude {action_magnitude}")
        #     action *= 0.0
        # elif action_magnitude > 1/eps:
        #     logging.warning(f"Fabrics: Avoiding large action with magnitude {action_magnitude}")
        #     action *= 0.0
        return action

class ParameterizedFabricPlanner(object):
    
    _dof: int
    _config: FabricPlannerConfig
    _problem_configuration : ProblemConfiguration
    leaves: Dict[str, Leaf]
    _ref_sign: int


    def __init__(self, dof: int, forward_kinematics: ForwardKinematics, **kwargs):
        self._dof = dof
        self._config = FabricPlannerConfig(**kwargs)
        self._forward_kinematics = forward_kinematics
        self.initialize_joint_variables()
        self.set_base_geometry()
        self._target_velocity = np.zeros(self._geometry.x().size()[0])
        self._ref_sign = 1
        self.leaves = {}

    """ INITIALIZING """

    def load_fabrics_configuration(self, fabrics_configuration: dict):
        self._config = FabricPlannerConfig(**fabrics_configuration)

    def initialize_joint_variables(self):
        q = ca.SX.sym("q", self._dof)
        qdot = ca.SX.sym("qdot", self._dof)
        self._variables = Variables(state_variables={"q": q, "qdot": qdot})

    def set_base_geometry(self):
        q = self._variables.position_variable()
        qdot = self._variables.velocity_variable()
        new_parameters, base_energy =  parse_symbolic_input(self._config.base_energy, q, qdot)
        print('basse_ energy', base_energy)
        self._variables.add_parameters(new_parameters)
        base_geometry = Geometry(h=ca.SX(np.zeros(self._dof)), var=self.variables)
        base_lagrangian = Lagrangian(base_energy, var=self._variables)
        self._geometry = WeightedGeometry(g=base_geometry, le=base_lagrangian)

    @property
    def variables(self) -> Variables:
        return self._variables

    @property
    def config(self) -> FabricPlannerConfig:
        return self._config

    """ ADDING COMPONENTS """

    def add_geometry(
        self, forward_map: DifferentialMap, lagrangian: Lagrangian, geometry: Geometry
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        weighted_geometry = WeightedGeometry(g=geometry, le=lagrangian)
        self.add_weighted_geometry(forward_map, weighted_geometry)

    def add_dynamic_geometry(
        self,
        forward_map: DifferentialMap,
        dynamic_map: DynamicDifferentialMap,
        geometry_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(geometry_map, DifferentialMap)
        assert isinstance(dynamic_map, DynamicDifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        weighted_geometry = WeightedGeometry(g=geometry, le=lagrangian, ref_names=dynamic_map.ref_names())
        pwg1 = weighted_geometry.pull(geometry_map)
        pwg2 = pwg1.dynamic_pull(dynamic_map)
        pwg3 = pwg2.pull(forward_map)
        self._geometry += pwg3

    def add_weighted_geometry(
        self, forward_map: DifferentialMap, weighted_geometry: WeightedGeometry
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(weighted_geometry, WeightedGeometry)
        pulled_geometry = weighted_geometry.pull(forward_map)
        self._geometry += pulled_geometry
        #self._refTrajs = joinRefTrajs(self._refTrajs, eg._refTrajs)
        self._variables = self._variables + pulled_geometry._vars

    def add_leaf(self, leaf: Leaf, prime_leaf: bool= False) -> None:
        if isinstance(leaf, GenericAttractor):
            self.add_forcing_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry(), prime_leaf)
        elif isinstance(leaf, GenericDynamicAttractor):
            self.add_dynamic_forcing_geometry(leaf.map(), leaf.dynamic_map(), leaf.lagrangian(), leaf.geometry(), leaf._xdot_ref, prime_leaf)
        elif isinstance(leaf, GenericGeometryLeaf):
            self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())
        elif isinstance(leaf, GenericDynamicGeometryLeaf):
            self.add_dynamic_geometry(leaf.map(), leaf.dynamic_map(), leaf.geometry_map(), leaf.lagrangian(), leaf.geometry())
        self.leaves[leaf._leaf_name] = leaf

    def get_leaves(self, leaf_names:list) -> List[Leaf]:
        leaves = []
        for leaf_name in leaf_names:
            if leaf_name not in self.leaves:
                error_message = f"Leaf with name {leaf_name} not in leaves.\n"
                error_message = f"Possible leaves are {list(self.leaves.keys())}."
                raise LeafNotFoundError(error_message)
            leaves.append(self.leaves[leaf_name])
        return leaves

    def add_forcing_geometry(
        self,
        forward_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
        prime_forcing_leaf: bool,
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        if not hasattr(self, '_forced_geometry'):
            self._forced_geometry = deepcopy(self._geometry)
        self._forced_geometry += WeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(forward_map)
        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map
        self._variables = self._variables + self._forced_geometry._vars
        self._geometry.concretize()
        self._forced_geometry.concretize(ref_sign=self._ref_sign)

    def add_dynamic_forcing_geometry(
        self,
        forward_map: DifferentialMap,
        dynamic_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
        target_velocity: ca.SX,
        prime_forcing_leaf: bool,
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(dynamic_map, DynamicDifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        assert isinstance(target_velocity, ca.SX)
        if not hasattr(self, '_forced_geometry'):
            self._forced_geometry = deepcopy(self._geometry)
        wg = WeightedGeometry(g=geometry, le=lagrangian)
        pwg = wg.dynamic_pull(dynamic_map)
        ppwg = pwg.pull(forward_map)
        self._forced_geometry += ppwg
        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map
        self._variables = self._variables + self._forced_geometry._vars
        self._target_velocity += ca.mtimes(ca.transpose(forward_map._J), target_velocity)
        self._ref_sign = -1
        self._geometry.concretize()
        self._forced_geometry.concretize(ref_sign=self._ref_sign)

    def set_execution_energy(self, execution_lagrangian: Lagrangian):
        assert isinstance(execution_lagrangian, Lagrangian)
        composed_geometry = Geometry(s=self._geometry)
        self._execution_lagrangian = execution_lagrangian
        self._execution_geometry = WeightedGeometry(
            g=composed_geometry, le=execution_lagrangian
        )
        self._execution_geometry.concretize()
        try:
            forced_geometry = Geometry(s=self._forced_geometry)
            self._forced_speed_controlled_geometry = WeightedGeometry(
                g=forced_geometry, le=execution_lagrangian
            )
            self._forced_speed_controlled_geometry.concretize()
        except AttributeError:
            logging.warning("No damping")

    def set_speed_control(self):
        x_psi = self._forced_variables.position_variable()
        dm_psi = self._forced_forward_map
        exLag = self._execution_lagrangian
        a_ex = ca.SX.sym("a_ex", 1)
        a_le = ca.SX.sym("a_le", 1)
        beta_expression = self.config.damper_beta
        eta_expression = self.config.damper_eta
        self._damper = Damper(beta_expression, eta_expression, x_psi, dm_psi, exLag._l)
        self._variables.add_parameters(self._damper.symbolic_parameters())

    def get_forward_kinematics(self, link_name, position_only: bool = True) -> ca.SX:
        if isinstance(link_name, ca.SX):
            return link_name
        fk = self._forward_kinematics.casadi(
                self._variables.position_variable(),
                link_name,
                position_only=position_only
            )
        return fk

    def add_capsule_sphere_geometry(
            self,
            obstacle_name: str,
            capsule_name: str,
            tf_capsule_origin: ca.SX,
            capsule_length: float
            ) -> None:
        tf_origin_center_0 = np.identity(4)
        tf_origin_center_0[2][3] = capsule_length / 2
        tf_center_0 = ca.mtimes(tf_capsule_origin, tf_origin_center_0)
        tf_origin_center_1 = np.identity(4)
        tf_origin_center_1[2][3] = - capsule_length / 2
        tf_center_1 = ca.mtimes(tf_capsule_origin, tf_origin_center_1)
        capsule_sphere_leaf = CapsuleSphereLeaf(
            self._variables,
            capsule_name,
            obstacle_name,
            tf_center_0[0:3,3],
            tf_center_1[0:3,3],
        )
        capsule_sphere_leaf.set_geometry(self.config.collision_geometry)
        capsule_sphere_leaf.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(capsule_sphere_leaf)

    def add_capsule_cuboid_geometry(
            self,
            obstacle_name: str,
            capsule_name: str,
            tf_capsule_origin: ca.SX,
            capsule_length: float
    ):
        tf_origin_center_0 = np.identity(4)
        tf_origin_center_0[2][3] = capsule_length / 2
        tf_center_0 = ca.mtimes(tf_capsule_origin, tf_origin_center_0)
        tf_origin_center_1 = np.identity(4)
        tf_origin_center_1[2][3] = - capsule_length / 2
        tf_center_1 = ca.mtimes(tf_capsule_origin, tf_origin_center_1)
        capsule_cuboid_leaf = CapsuleCuboidLeaf(
            self._variables,
            capsule_name,
            obstacle_name,
            tf_center_0[0:3,3],
            tf_center_1[0:3,3],
        )
        capsule_cuboid_leaf.set_geometry(self.config.collision_geometry)
        capsule_cuboid_leaf.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(capsule_cuboid_leaf)

    def add_spherical_obstacle_geometry(
            self,
            obstacle_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            ) -> None:
        """
        Add a spherical obstacle geometry to the fabrics planner.

        Parameters
        ----------
        obstacle_name : str
            The name of the obstacle to be added.
        collision_link_name : str
            The name of the robot's collision link that the obstacle is associated with.
        forward_kinematics : ca.SX
            The forward kinematics expression representing the obstacle's position.

        Returns
        -------
        None

        Notes
        -----
        - The `forward_kinematics` should be a symbolic
            expression using CasADi SX for the obstacle's position.
        - The `collision_geometry` and `collision_finsler` configurations
            must be set before adding the obstacle.
        - After adding the obstacle, it will be included in the robot's
            configuration and affect its motion planning.

        Example
        -------
        obstacle_name = "Sphere_Obstacle"
        collision_link_name = "Link1"
        forward_kinematics = ca.SX([fk_x, fk_y, fk_z])
        robot.add_spherical_obstacle_geometry(
            obstacle_name,
            collision_link_name,
            forward_kinematics
        )
        """
        geometry = ObstacleLeaf(
            self._variables,
            forward_kinematics,
            obstacle_name,
            collision_link_name,
        )
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)

    def add_dynamic_spherical_obstacle_geometry(
            self,
            obstacle_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            reference_parameters: dict,
            dynamic_obstacle_dimension: int = 3,
            ) -> None:
        geometry = DynamicObstacleLeaf(
            self._variables,
            forward_kinematics[0:dynamic_obstacle_dimension],
            obstacle_name,
            collision_link_name,
            reference_parameters=reference_parameters
        )
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)

    def add_plane_constraint(
            self,
            constraint_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            ) -> None:

        geometry = PlaneConstraintGeometryLeaf(
            self._variables,
            constraint_name,
            collision_link_name,
            forward_kinematics,
        )
        geometry.set_geometry(self.config.geometry_plane_constraint)
        geometry.set_finsler_structure(self.config.finsler_plane_constraint)
        self.add_leaf(geometry)

    def add_cuboid_obstacle_geometry(
            self,
            obstacle_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            ) -> None:

        geometry = SphereCuboidLeaf(
            self._variables,
            forward_kinematics,
            obstacle_name,
            collision_link_name,
        )
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)
    
    def add_esdf_geometry(
            self,
            collision_link_name: str,
            ) -> None:
        fk = self.get_forward_kinematics(collision_link_name)
        geometry = ESDFGeometryLeaf(self._variables, collision_link_name, fk)
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)

    def add_spherical_self_collision_geometry(
            self,
            collision_link_1: str,
            collision_link_2: str,
            ) -> None:
        fk_1 = self.get_forward_kinematics(collision_link_1)
        fk_2 = self.get_forward_kinematics(collision_link_2)
        fk = fk_2 - fk_1
        if is_sparse(fk):
            message = (
                    f"Expression {fk} for links {collision_link_1} "
                    "and {collision_link_2} is sparse and thus skipped."
            )
            logging.warning(message.format_map(locals()))
        geometry = SelfCollisionLeaf(self._variables, fk, collision_link_1, collision_link_2)
        geometry.set_geometry(self.config.self_collision_geometry)
        geometry.set_finsler_structure(self.config.self_collision_finsler)
        self.add_leaf(geometry)

    def add_limit_geometry(
            self,
            joint_index: int,
            limits: list,
            ) -> None:
        lower_limit_geometry = LimitLeaf(self._variables, joint_index, limits[0], 0)
        lower_limit_geometry.set_geometry(self.config.limit_geometry)
        lower_limit_geometry.set_finsler_structure(self.config.limit_finsler)
        upper_limit_geometry = LimitLeaf(self._variables, joint_index, limits[1], 1)
        upper_limit_geometry.set_geometry(self.config.limit_geometry)
        upper_limit_geometry.set_finsler_structure(self.config.limit_finsler)
        self.add_leaf(lower_limit_geometry)
        self.add_leaf(upper_limit_geometry)

    def load_problem_configuration(self, problem_configuration: ProblemConfiguration):
        self._problem_configuration = ProblemConfiguration(**problem_configuration)
        for obstacle in self._problem_configuration.environment.obstacles:
            self._variables.add_parameters(obstacle.sym_parameters)

        self.set_collision_avoidance()
        #self.set_self_collision_avoidance()
        self.set_joint_limits()
        if self._config.forcing_type in ['forced', 'speed-controlled', 'forced-energized']:
            self.set_goal_component(self._problem_configuration.goal_composition)
        if self._config.forcing_type in ['speed-controlled', 'execution-energy', 'forced-energized']:
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
        if self._config.forcing_type in ['speed-controlled']:
            self.set_speed_control()

    def set_joint_limits(self):
        limits = np.zeros((self._dof, 2))
        limits[:, 0] = self._problem_configuration.joint_limits.lower_limits
        limits[:, 1] = self._problem_configuration.joint_limits.upper_limits
        limits = limits.tolist()
        for joint_index in range(len(limits)):
            self.add_limit_geometry(joint_index, limits[joint_index])

    def set_self_collision_avoidance(self) -> None:
        if not self._problem_configuration.robot_representation.self_collision_pairs:
            return
        for link_name,  paired_links_names in self._problem_configuration.robot_representation.self_collision_pairs.items():
            link = self._problem_configuration.robot_representation.collision_links[link_name]
            for paired_link_name in paired_links_names:
                paired_link = self._problem_configuration.robot_representation.collision_links[paired_link_name]
                if isinstance(link, Sphere) and isinstance(paired_link, Sphere):
                    self.add_spherical_self_collision_geometry(
                            paired_link_name,
                            link_name,
                    )


    def set_collision_avoidance(self) -> None:
        if not self._problem_configuration.robot_representation.collision_links:
            return
        for link_name, collision_link in self._problem_configuration.robot_representation.collision_links.items():
            fk = self.get_forward_kinematics(link_name, position_only=False)
            if fk.shape == (3, 3):
                fk_augmented = ca.SX.eye(4)
                fk_augmented[0:2, 0:2] = fk[0:2, 0:2]
                fk_augmented[0:2, 3] = fk[0:2, 2]
                fk = fk_augmented
            if fk.shape == (4, 4) and is_sparse(fk[0:3, 3]):
                message = (
                        f"Expression {fk[0:3, 3]} for link {link_name} "
                        "is sparse and thus skipped."
                )
                logging.warning(message.format_map(locals()))
                continue
            elif fk.shape != (4, 4) and is_sparse(fk):
                message = (
                        f"Expression {fk} for link {link_name} "
                        "is sparse and thus skipped."
                )
                logging.warning(message.format_map(locals()))
                continue
            collision_link.set_origin(fk)
            self._variables.add_parameters(collision_link.sym_parameters)
            self._variables.add_parameters_values(collision_link.parameters)
            for obstacle in self._problem_configuration.environment.obstacles:
                distance = collision_link.distance(obstacle)
                leaf_name = f"{link_name}_{obstacle.name}_leaf"
                leaf = AvoidanceLeaf(self._variables, leaf_name, distance)
                leaf.set_geometry(self.config.collision_geometry)
                leaf.set_finsler_structure(self.config.collision_finsler)
                self.add_leaf(leaf)
            """
            for i in range(self._problem_configuration.environment.number_spheres['dynamic']):
                obstacle_name = f"obst_dynamic_{i}"
                if isinstance(collision_link, Sphere):
                    self.add_dynamic_spherical_obstacle_geometry(
                            obstacle_name,
                            link_name,
                            fk,
                            reference_parameter_list[i],
                            dynamic_obstacle_dimension=dynamic_obstacle_dimension,
                    )
            for i in range(self._problem_configuration.environment.number_planes):
                constraint_name = f"constraint_{i}"
                if isinstance(collision_link, Sphere):
                    self.add_plane_constraint(constraint_name, link_name, fk)

            for i in range(self._problem_configuration.environment.number_cuboids['static']):
                obstacle_name = f"obst_cuboid_{i}"
                if isinstance(collision_link, Sphere):
                    self.add_cuboid_obstacle_geometry(obstacle_name, link_name, fk)
            """




    def set_components(
        self,
        collision_links: Optional[list] = None,
        self_collision_pairs: Optional[dict] = None,
        collision_links_esdf: Optional[list] = None,
        goal: Optional[GoalComposition] = None,
        limits: Optional[list] = None,
        number_obstacles: int = 1,
        number_dynamic_obstacles: int = 0,
        number_obstacles_cuboid: int = 0,
        number_plane_constraints: int = 0,
        dynamic_obstacle_dimension: int = 3,
    ):
        collision_links = collision_links or []
        collision_links_esdf = collision_links_esdf or []
        self_collision_pairs = self_collision_pairs or {}

        reference_parameter_list = []

        
        # for i in range(number_dynamic_obstacles):
        #     reference_parameters = {
        #         f"x_obst_dynamic_{i}": ca.SX.sym(f"x_obst_dynamic_{i}", dynamic_obstacle_dimension),
        #         f"xdot_obst_dynamic_{i}": ca.SX.sym(f"xdot_obst_dynamic_{i}", dynamic_obstacle_dimension),
        #         f"xddot_obst_dynamic_{i}": ca.SX.sym(f"xddot_obst_dynamic_{i}", dynamic_obstacle_dimension),
        #     }
        #     reference_parameter_list.append(reference_parameters)
        # for collision_link in collision_links:
        #     fk = self.get_forward_kinematics(collision_link)
        #     if is_sparse(fk):
        #         message = (
        #                 f"Expression {fk} for link {collision_link} "
        #                 "is sparse and thus skipped."
        #         )
        #         logging.warning(message.format_map(locals()))
        #         continue
        #     for i in range(number_obstacles):
        #         obstacle_name = f"obst_{i}"
        #         print("obstacle_name: ", obstacle_name)
        #         self.add_spherical_obstacle_geometry(obstacle_name, collision_link, fk)
        #     for i in range(number_dynamic_obstacles):
        #         obstacle_name = f"obst_dynamic_{i}"
        #         print("dynamic_obstacle_name: ", obstacle_name)
        #         self.add_dynamic_spherical_obstacle_geometry(
        #                 obstacle_name,
        #                 collision_link,
        #                 fk,
        #                 reference_parameter_list[i],
        #                 dynamic_obstacle_dimension=dynamic_obstacle_dimension,
        #         )
        #     for i in range(number_plane_constraints):
        #         constraint_name = f"constraint_{i}"
        #         print("constraint_name: ", constraint_name, collision_link)
        #         self.add_plane_constraint(constraint_name, collision_link, fk)

        #     for i in range(number_obstacles_cuboid):
        #         obstacle_name = f"obst_cuboid_{i}"
        #         print("obstacles_cuboid_name: ", obstacle_name)
        #         self.add_cuboid_obstacle_geometry(obstacle_name, collision_link, fk)


        # for collision_link in collision_links_esdf:
        #     self.add_esdf_geometry(collision_link)

        # for self_collision_key, self_collision_list in self_collision_pairs.items():
        #     for self_collision_link in self_collision_list:
        #         self.add_spherical_self_collision_geometry(
        #                 self_collision_link,
        #                 self_collision_key,
        #         )

        # if limits:
        #     for joint_index in range(len(limits)):
        #         self.add_limit_geometry(joint_index, limits[joint_index])
        # execution_energy = ExecutionLagrangian(self._variables)
        # self.set_execution_energy(execution_energy)
        print("variables", self._variables)
        if goal:
            self.set_goal_component(goal)
            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()

    def get_differential_map(self, sub_goal_index: int, sub_goal: SubGoal):
        if sub_goal.type() == 'staticJointSpaceSubGoal':
            return self._variables.position_variable()[sub_goal.indices()]
        else:
            fk_child = self.get_forward_kinematics(sub_goal.child_link())
            try:
                fk_parent = self.get_forward_kinematics(sub_goal.parent_link())
            except LinkNotInURDFError as e:
                fk_parent = ca.SX(np.zeros(3))
            angles = sub_goal.angle()

            if angles and isinstance(angles, list) and len(angles) == 4:
                logging.warning(
                    "Subgoal attribute 'angle' deprecated. " \
                    +"Remove the goal attribute angle and rotate the" \
                    +"position before passing it into"\
                    +"compute_action."
                )
                angles = ca.SX.sym(f"angle_goal_{sub_goal_index}", 3, 3)
                self._variables.add_parameter(f'angle_goal_{sub_goal_index}', angles)
                # rotation
                R = compute_rotation_matrix(angles)
                fk_child = ca.mtimes(R, fk_child)
                fk_parent = ca.mtimes(R, fk_parent)
            elif angles:
                logging.warning(
                    "Subgoal attribute 'angle' deprecated. " \
                    +"Remove the goal attribute angle and rotate the" \
                    +"position before passing it into"\
                    +"compute_action."
                )
                R = compute_rotation_matrix(angles)
                fk_child = ca.mtimes(R, fk_child)
                fk_parent = ca.mtimes(R, fk_parent)
            return fk_child[sub_goal.indices()] - fk_parent[sub_goal.indices()]



    def set_goal_component(self, goal: GoalComposition):
        # Adds default attractor
        for j, sub_goal in enumerate(goal.sub_goals()):
            fk_sub_goal = self.get_differential_map(j, sub_goal)
            if is_sparse(fk_sub_goal):
                raise ExpressionSparseError()
            if sub_goal.type() in ["analyticSubGoal", "splineSubGoal"]:
                attractor = GenericDynamicAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            else:
                self._variables.add_parameter(f'x_goal_{j}', ca.SX.sym(f'x_goal_{j}', sub_goal.dimension()))
                attractor = GenericAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            attractor.set_potential(self.config.attractor_potential)
            attractor.set_metric(self.config.attractor_metric)
            self.add_leaf(attractor, prime_leaf=sub_goal.is_primary_goal())





    def concretize(self, mode='acc', time_step=None):

        self._mode = mode

        self._geometry.concretize()
        if self._config.forcing_type in ['speed-controlled']:
            eta = self._damper.substitute_eta()
            a_ex = (
                eta * self._execution_geometry._alpha
                + (1 - eta) * self._forced_speed_controlled_geometry._alpha
            )
            beta_subst = self._damper.substitute_beta(-a_ex, -self._geometry._alpha)
            xddot = self._forced_geometry._xddot - (a_ex + beta_subst) * (
                self._geometry.xdot()
                - ca.mtimes(self._forced_geometry.Minv(), self._target_velocity)
            )
            #xddot = self._forced_geometry._xddot
        if mode == 'acc':
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables, {"action": xddot, "eta": eta,"a_ex": a_ex, "beta": beta_subst,"exe_alpha": self._execution_geometry._alpha, \
                                         "forced_xddot": self._forced_geometry._xddot,"force_alpha": self._forced_speed_controlled_geometry._alpha}
            )
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables, {"action": xddot}
            )
            

    def serialize(self, file_name: str):
        """
        Serializes the fabric planner.

        The file can be loaded using the serialized_planner.
        Essentially, only the casadiFunctionWrapper is serialized using
        pickle.
        """
        self._funs.serialize(file_name)

    def export_as_xml(self, file_name: str):
        """
        Exports the pure casadi function in xml format.

        The generated file can be loaded in python, cpp or Matlab.
        You can use that using the syntax ca.Function.load(file_name).
        Note that passing arguments as dictionary is not supported then.
        """
        function = self._funs.function()
        function.save(file_name)

    def export_as_c(self, file_name: str):
        """
        Export the planner as c source code.
        """
        function = self._funs.function()
        function.generate(file_name)
 
    """ RUNTIME METHODS """

    def compute_action(self, **kwargs):
        """
        Computes action based on the states passed.

        The variables passed are the joint states, and the goal position.
        The action is nullified if its magnitude is very large or very small.
        """
        # panda_limits = [
        #     [-2.8973, 2.8973],
        #     [-1.7628, 1.7628],
        #     [-2.8973, 2.8973],
        #     [-3.0718, -0.0698],
        #     [-2.8973, 2.8973],
        #     [-0.0175, 3.7525],
        #     [-2.8973, 2.8973]
        # ]
        # panda_limits = torch.tensor(panda_limits, dtype=torch.float64)
        # panda_limits = np.array(panda_limits, dtype=np.float64)
        # lower = kwargs["q"]-panda_limits[:,0]
        # upper = panda_limits[:,1]- kwargs["q"]
        # limit_ex = self._config.limit_geometry
        
        # attractor = self.leaves['goal_0_leaf']
        # attractor._map.concretize()
        # x, xdot, J, Jdot = attractor._map.forward(**kwargs)
        
        # M_subst = self._pulled_attractor._M_subst
        # f_subst = self._pulled_attractor._f_subst
        # l_subst = self._pulled_attractor._l_subst
        # subst = CasadiFunctionWrapper(
        #         "subst", self.variables, {"M_subst": M_subst, "f_subst": f_subst, "l_subst" : l_subst}
        #     )
        # s= subst.evaluate(**kwargs)  
        # M = s['M_subst']
        # f = s['f_subst']

        # l = s["l_subst"]
        # # M, Minv, f, forced_xddot, alpha, frac, lag_M, lag_f= self._forced_geometry.evaluate(**kwargs)

        # self._forced_geometry.concretize()
        # self._execution_geometry.concretize()
        # self._pulled_attractor.concretize()
        # # print("leaves", self.leaves)
        # for i in range(self._dof):
        #     limit = self.leaves[f"limit_joint_{i}_{0}_leaf"]

        #     if i==0:
        #         limit_geom = WeightedGeometry(g=limit._geo, le=limit._lag).pull(limit.map())
        #     else:
        #         limit_geom += WeightedGeometry(g=limit._geo, le=limit._lag).pull(limit.map())
        # limit_geom.concretize()

        # base_M, base_Minv, base_f, base_xddot, base_alpha, base_frac, base_lag_l, base_lag_M, base_lag_f= self._geometry.evaluate(**kwargs)
        # limit_M, limit_Minv, limit_f, limit_xddot, limit_alpha, limit_frac, limit_lag_l, limit_lag_M, limit_lag_f= limit_geom.evaluate(**kwargs)
        # exe_M, exe_Minv, exe_f, exe_xddot, exe_alpha, exe_frac, exe_lag_l, exe_lag_M, exe_lag_f= self._execution_geometry.evaluate(**kwargs)
        # force_M, force_Minv, force_f, force_xddot, force_alpha, force_frac, force_lag_l, force_lag_M, force_lag_f= self._forced_geometry.evaluate(**kwargs)
        # attractor_M, attractor_Minv, attractor_f, attractor_xddot, attractor_alpha, attractor_frac, attractor_lag_l, attractor_lag_M, attractor_lag_f= self._pulled_attractor.evaluate(**kwargs)

        # print("base M1", base_M)
        # print("base f1", base_f)
        # print("base xddot1", base_xddot)
        # print("base alpha1", base_alpha)
        # print("base frac1", base_frac)
        # print("base lag_M1", base_lag_M)
        # print("base lag_f1", base_lag_f)

        # print("limit M1", limit_M)
        # print("limit f1", limit_f)
        # print("limit xddot1", limit_xddot)
        # print("limit alpha1", limit_alpha)
        # print("limit frac1", limit_frac)
        # print("limit lag_l1", limit_lag_l)
        # print("limit lag_M1", limit_lag_M)
        # print("limit lag_f1", limit_lag_f)

        # print("exe M1", exe_M)
        # print("exe f1", exe_f)
        # print("exe xddot1", exe_xddot)
        # print("exe alpha1", exe_alpha)
        # print("exe frac1", exe_frac)
        # print("exe lag_l1", exe_lag_l)
        # print("exe lag_M1", exe_lag_M)
        # print("exe lag_f1", exe_lag_f)

        # print("attractor M1", attractor_M)
        # print("attractor f1", attractor_f)
        # print("attractor xddot1", attractor_xddot)
        # print("attractor alpha1", attractor_alpha)
        # print("attractor frac1", attractor_frac)
        # print("attractor lag_l1", attractor_lag_l)
        # print("attractor lag_M1", attractor_lag_M)
        # print("attractor lag_f1", attractor_lag_f)


        # print("force M1", force_M)
        # print("force f1", force_f)
        # print("force xddot1", force_xddot)
        # print("force alpha1", force_alpha)
        # print("force frac1", force_frac)
        # print("force lag_l1", force_lag_l)
        # print("force lag_M1", force_lag_M)
        # print("force lag_f1", force_lag_f)

        # print("pulled f1", base_f)
        # print("pulled xddot1", xddot)
        # print("force f1", force_f)
        # # print("h1", force_h)
        # print("force xddot1", force_xddot)
        # print("alpha1", force_alpha)
        # print("frac1", force_frac)
        # print("lag_M1", lag_M)
        # print("lag_f1", lag_f)
        # print("M1", M)
        # print("f1", f)
        # print("xddot1", xddot)
        # print("alpha1", alpha)
        # print("frac1", frac)
        # print("l_subst1", l)
        # print("lag_M1", lag_M)
        # print("lag_f1", lag_f)
        # print("l_subst1", l)
        # print("S", S)
        # print("Mcond", Mcond)
        # print("M_subst1", M)
        # print("f_subst1", f)
        
        # # Jt = np.transpose(J)
        # not_pull_attractor = WeightedGeometry(g=attractor._geo, le= attractor._lag)
        # not_pull_attractor.computeAlpha()
        # attractor_M = ca.substitute(not_pull_attractor.M(), attractor._geo.x(), attractor._map._phi)
        # attractor_M = ca.substitute(attractor_M, attractor._geo.xdot(), attractor._map._phidot)
        # attractor_alpha = ca.substitute(not_pull_attractor._alpha, attractor._geo.x(), attractor._map._phi)
        # attractor_alpha = ca.substitute(attractor_alpha, attractor._geo.xdot(), attractor._map._phidot)

        # attractor_f = ca.substitute(not_pull_attractor.f(), attractor._geo.x(), attractor._map._phi)
        # attractor_f = ca.substitute(attractor_f, attractor._geo.xdot(), attractor._map._phidot)
        # attractor_h = ca.substitute(not_pull_attractor.h(), attractor._geo.x(), attractor._map._phi)
        # attractor_h = ca.substitute(attractor_h, attractor._geo.xdot(), attractor._map._phidot)
        # attractor_frac = ca.substitute(not_pull_attractor.frac, attractor._geo.x(), attractor._map._phi)
        # attractor_frac = ca.substitute(attractor_frac, attractor._geo.xdot(), attractor._map._phidot)
        # attractor_xddot = ca.substitute(not_pull_attractor._xddot, attractor._geo.x(), attractor._map._phi)
        # attractor_xddot = ca.substitute(attractor_xddot, attractor._geo.xdot(), attractor._map._phidot)
        # attractor_le_M = ca.substitute(not_pull_attractor._le._S._M, attractor._geo.x(), attractor._map._phi)
        # attractor_le_M = ca.substitute(attractor_le_M, attractor._geo.xdot(), attractor._map._phidot)
        # attractor_le_f = ca.substitute(not_pull_attractor._le._S.f(), attractor._geo.x(), attractor._map._phi)
        # attractor_le_f = ca.substitute(attractor_le_f, attractor._geo.xdot(), attractor._map._phidot)

        # func = CasadiFunctionWrapper(
        #         "asdf", self.variables, {"attractor_M": attractor_M, "attractor_f": attractor_f, "attractor_frac" : attractor_frac,
        #                                 "attractor_alpha": attractor_alpha, "attractor_h" : attractor_h,
        #                                  "attractor_xddot": attractor_xddot, "attractor_le_M": attractor_le_M, "attractor_le_f": attractor_le_f}
        #     )
        # evaluations = func.evaluate(**kwargs)
        # attractor_M = evaluations["attractor_M"]
        # attractor_f = evaluations["attractor_f"]
        # attractor_h = evaluations["attractor_h"]
        # attractor_alpha= evaluations["attractor_alpha"]
        # attractor_frac = evaluations["attractor_frac"]
        # attractor_xddot = evaluations["attractor_xddot"]
        # attractor_le_M= evaluations["attractor_le_M"]
        # attractor_le_f = evaluations["attractor_le_f"]
        # print("M1", evaluations["attractor_M"])
        # print("f1", evaluations["attractor_f"])
        # print("h1", evaluations["attractor_h"])
        # print("alpha1", evaluations["attractor_alpha"])
        # print("frac1", evaluations["attractor_frac"])
        # print("xddot1", evaluations["attractor_xddot"])
        # print("le_M1", evaluations["attractor_le_M"])
        # print("le_f1", evaluations["attractor_le_f"])

        # frac_subst = ca.substitute(not_pull_attractor.frac, attractor._geo.x(),attractor._map._phi)
        # frac_subst = ca.substitute(frac_subst, attractor._geo.xdot(),attractor._map._phidot)
        # lag_M_subst = ca.substitute(not_pull_attractor._le._S._M, attractor._geo.x(),attractor._map._phi)
        # lag_M_subst = ca.substitute(lag_M_subst, attractor._geo.x(),attractor._map._phidot)
        # alpha_subst = ca.substitute(not_pull_attractor._alpha, attractor._geo.x(),attractor._map._phi)
        # alpha_subst = ca.substitute(alpha_subst, attractor._geo.xdot(),attractor._map._phidot)
        # f_subst = ca.substitute(not_pull_attractor.f(), attractor._geo.x(),attractor._map._phi)
        # f_subst = ca.substitute(f_subst, attractor._geo.xdot(),attractor._map._phidot)
        # h = self._pulled_attractor.h()
        # frac = self._pulled_attractor.frac
        # lag_M = self._pulled_attractor._le._S._M
        # alpha = self._pulled_attractor._alpha
        # f = self._pulled_attractor.f()
        # subst = CasadiFunctionWrapper(
        #         "subst", self.variables, {"h_subst": h, "frac_subst": frac, "lag_M_subst": lag_M, "alpha_subst": alpha,"f_subst": f}
        #     )
        # h  = subst.evaluate(**kwargs)['h_subst']
        # frac  = subst.evaluate(**kwargs)['frac_subst']
        # lag_M  = subst.evaluate(**kwargs)['lag_M_subst']
        # alpha  = subst.evaluate(**kwargs)['alpha_subst']
        # f  = subst.evaluate(**kwargs)['f_subst']
        # [M, Minv, f, xddot, alpha, frac, lag_f] = self._forced_geometry.evaluate(**kwargs)
        # print("frac1", frac)
        # print("M1", M)
        # print("alpha1", alpha)
        # print("xddot1", xddot)
        # print("f1", f)
        # attractor._map.concretize()
        # x, xdot, J, Jdot = attractor._map.forward(**kwargs)
        # print("x1", x)
        # print("xdot1", xdot)
        # print("l_subst1", l)
        # print("M_subst1", M)
        # print("f_subst1", f)

        # print("Jdot1", Jdot)
        evaluations = self._funs.evaluate(**kwargs) 

        action = evaluations["action"]
        # forced_xddot = evaluations["forced_xddot"]
        # print("forced_xddot1", forced_xddot)
        # Debugging
        # eta = evaluations['eta']
        # a_ex = evaluations['a_ex']
        # beta = evaluations['beta']
        # # exe_alpha = evaluations['exe_alpha']
        # force_alpha = evaluations['force_alpha']
        # logging.debug(f"a_ex: {evaluations['a_ex']}")
        # logging.debug(f"alhpa_forced_geometry: {evaluations['alpha_forced_geometry']}")
        # logging.debug(f"alpha_geometry: {evaluations['alpha_geometry']}")
        # logging.debug(f"beta : {evaluations['beta']}")
        # print("eta1", eta)
        # print("a_ex1", a_ex)
        # print("beta1", beta)
        # print("alpha1", force_alpha)
        # print("base alpha1", base_alpha)

        action_magnitude = np.linalg.norm(action)
        if action_magnitude < eps:
            # logging.warning(f"Fabrics: Avoiding small action with magnitude {action_magnitude}")
            action *= 0.0
        elif action_magnitude > 1/eps:
            logging.warning(f"Fabrics: Avoiding large action with magnitude {action_magnitude}")
            action *= 0.0
        return action


