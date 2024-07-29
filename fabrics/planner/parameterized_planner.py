import logging
from copy import deepcopy
from typing import Dict, List, Optional

import casadi as ca
import deprecation
import numpy as np

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
                                                GenericGeometryLeaf, LimitLeaf,
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
        base_h = TorchFunctionWrapper(expression=lambda : torch.zeros(self._dof,dtype=torch.float64), variables=self._variables.toTorch(), ex_input=[])
        base_geometry = TorchGeometry(h = base_h, var=self._variables.toTorch())
        base_lagrangian = TorchLagrangian(base_energy, var=self._variables.toTorch())
        self._geometry = TorchWeightedGeometry(g=base_geometry, le=base_lagrangian)
        self.base_energy = base_energy
    
    """ ADDING COMPONENTS"""
    def add_geometry(
        self, forward_map: TorchDifferentialMap, lagrangian: TorchLagrangian, geometry: Geometry
    ) -> None:
        weighted_geometry =TorchWeightedGeometry(g=geometry, le=lagrangian)
        self.add_weighted_geometry(forward_map, weighted_geometry)
    
    def add_weighted_geometry(
        self, forward_map: TorchDifferentialMap, weighted_geometry: TorchWeightedGeometry
    ) -> None:
        pulled_geometry = weighted_geometry.pull(forward_map)
        self._geometry += pulled_geometry
        self._variables = self._variables + pulled_geometry._vars
    
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
    
    def add_leaf(self, leaf: Leaf, prime_leaf: bool= False) -> None:
        if isinstance(leaf, TorchGenericAttractor):
            self.add_forcing_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry(), prime_leaf)
        # elif isinstance(leaf, TorchGenericGeometryLeaf):
        #     self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())
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
            attractor.set_potential(self._config.attractor_potential)
            attractor.set_metric(self._config.attractor_metric)
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
        limits: Optional[list] = None,
        number_obstacles: int = 1,
        number_dynamic_obstacles: int = 0,
        number_obstacles_cuboid: int = 0,
        number_plane_constraints: int = 0,
        dynamic_obstacle_dimension: int = 3,
    ):
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
        attractor = self.leaves['goal_0_leaf']
        not_pulled_geom = TorchWeightedGeometry(g=attractor.geometry(), le=attractor.lagrangian())
        pulled_geom = not_pulled_geom.pull(attractor.map())
        # self._forced_geometry._xddot(**kwargs)
        # self._forced_geometry._Minv(**kwargs)
        attractor_M = not_pulled_geom._M.lowerLeaf(attractor.map())
        attractor_frac =  not_pulled_geom.frac.lowerLeaf(attractor.map())
        attractor_alpha =  not_pulled_geom._alpha.lowerLeaf(attractor.map())
        attractor_f =  not_pulled_geom._f.lowerLeaf(attractor.map())
        attractor_le_M = not_pulled_geom._le._S._M.lowerLeaf(attractor.map())
        attractor_le_f =  not_pulled_geom._le._S._f.lowerLeaf(attractor.map())
        attractor_xddot =  not_pulled_geom._xddot.lowerLeaf(attractor.map())
        # attractor_frac =  self._forced_geometry.frac
        # attractor_xddot =  self._forced_geometry._xddot
        # attractor_alpha =  self._forced_geometry._alpha
        # attractor_f =  self._forced_geometry._f
        # le_M = self._forced_geometry._le._S._M
        # attractor_lag_f =  self._forced_geometry._le._S._f
        M = self._forced_geometry._M(**kwargs)
        f = self._forced_geometry._f(**kwargs)
        xddot = self._forced_geometry._xddot(**kwargs)
        # alpha = self._forced_geometry._alpha(**kwargs)
        frac = self._forced_geometry.frac(**kwargs)
        # le_M = self._forced_geometry._le._S._M(**kwargs)
        # le_f = self._forced_geometry._le._S._f(**kwargs)

        # print("M2:", M)
        # print("f2:", f)
        # print("xddot2:", xddot)
        # # print("alpha2:", alpha)
        # print("frac2:", frac)
        # print("le_M:", le_M)
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
        q= self._variables._state_variable_names[0]
        qdot= self._variables._state_variable_names[1]
        l_subst = pulled_geom._l_subst
        dL_dxdot = l_subst.grad(qdot)
        dL_dx = l_subst.grad(q)
        d2L_dxdxdot = dL_dx.grad(qdot)
        d2L_dxdot2 = dL_dxdot.grad(qdot)

        F = d2L_dxdxdot
        # F(**kwargs)
        f_e = -dL_dx
        M = d2L_dxdot2
        f = F.transpose() @ kwargs[qdot] + f_e

        print("M2", pulled_geom._M(**kwargs))
        print("f2", pulled_geom._f(**kwargs))
        print("xddot2", pulled_geom._xddot(**kwargs))
        print("frac2", pulled_geom.frac(**kwargs))
        print("l_subst2", pulled_geom._l_subst(**kwargs))
        print("alpha2", pulled_geom._alpha(**kwargs))
        print("le_M2", pulled_geom._le._S._M(**kwargs))
        print("le_f2", f(**kwargs))

        # print("baseM2", pulled_geom._M(**kwargs))
        # print("basef2", pulled_geom._f(**kwargs))
        # print("basexddot2", pulled_geom._xddot(**kwargs))
        # print("basefrac2", pulled_geom.frac(**kwargs))
        # print("basel_subst2", pulled_geom._l_subst(**kwargs))
        # print("basealpha2", pulled_geom._alpha(**kwargs))
        # print("basele_M", pulled_geom._le._S._M(**kwargs))
        # print("basele_f", f(**kwargs))

        # print("forced_xddot2", forced_xddot)
        # print("a_ex", self.a_ex(**kwargs))
        # print("beat", self.beta_subst(**kwargs))
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
        action = self._xddot(**kwargs)
        action_magnitude = torch.linalg.norm(action, dim=-1)

        if action_magnitude < eps:
            # logging.warning(f"Fabrics: Avoiding small action with magnitude {action_magnitude}")
            action *= 0.0
        elif action_magnitude > 1/eps:
            logging.warning(f"Fabrics: Avoiding large action with magnitude {action_magnitude}")
            action *= 0.0
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

    def test_init_function(self, **kwargs):
        # print("base energy : ", self._geometry._le._l(**kwargs))
        print("q", kwargs['q'])
        print("qdot", kwargs['qdot'])
        print("leavs", self.get_leaves(["goal_0_leaf"]))
        # attractor = self.leaves['goal_0_leaf']
        # attractor._lag._S.concretize()
        M, f, _, alpha = self._geometry.evaluate(**kwargs)
        print("pulled M",  M)
        print("pulled f", f)
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
        self._base_energy = base_energy
        self._variables.add_parameters(new_parameters)
        base_geometry = Geometry(h=ca.SX(np.zeros(self._dof)), var=self._variables)
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
        self._pulled_attractor = WeightedGeometry(g=geometry,le=lagrangian).pull(forward_map)
        
        self._forced_geometry += WeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(forward_map)
        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map

        self._variables = self._variables + self._forced_geometry._vars
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

    def get_forward_kinematics(self, link_name, position_only: bool = True) -> ca.SX:
        if isinstance(link_name, ca.SX):
            return link_name
        fk = self._forward_kinematics.casadi(
                self._variables.position_variable(),
                link_name,
                position_only=position_only
            )
        return fk

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
        # print("----------------variables init", self._variables)
        if goal:
            
            self.set_goal_component(goal)
            # print("---------------variables after set_goal", self._variables)

            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # print("----------------variables after set_execution energy", self._variables)

            # Sets speed control
            self.set_speed_control()
            # print("---------------variables after set_speed_control", self._variables)


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



    def set_goal_component(self, goal: GoalComposition):
        # Adds default attractor
        for j, sub_goal in enumerate(goal.sub_goals()):
            fk_sub_goal = self.get_differential_map(j, sub_goal)
            if is_sparse(fk_sub_goal):
                raise ExpressionSparseError()
            # if sub_goal.type() in ["analyticSubGoal", "splineSubGoal"]:
            #     attractor = GenericDynamicAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            # else:
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
                "funs", self.variables, {"action": xddot, "a_ex": a_ex, "beta": beta_subst,"exe_alpha": self._execution_geometry._alpha, \
                                         "forced_xddot": self._forced_geometry._xddot,"force_alpha": self._forced_speed_controlled_geometry._alpha}
            )
            # self._funs = CasadiFunctionWrapper(
            #     "funs", self.variables, {"action": xddot}
            # )
            

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
        self._forced_geometry.concretize()
        # self._execution_geometry.concretize()
        self._pulled_attractor.concretize()
        M_subst = self._pulled_attractor._M_subst
        f_subst = self._pulled_attractor._f_subst
        l_subst = self._pulled_attractor._l_subst
        subst = CasadiFunctionWrapper(
                "subst", self.variables, {"M_subst": M_subst, "f_subst": f_subst, "l_subst" : l_subst}
            )
        s= subst.evaluate(**kwargs)  
        # M = s['M_subst']
        # f = s['f_subst']
        l = s["l_subst"]
        # M, Minv, f, forced_xddot, alpha, frac, lag_M, lag_f= self._forced_geometry.evaluate(**kwargs)
        M, Minv, f, xddot, alpha, frac, lag_M, lag_f= self._pulled_attractor.evaluate(**kwargs)
        # exe_M, exe_Minv, exe_f, _, exe_alpha, exe_frac, exe_lag_f= self._execution_geometry.evaluate(**kwargs)
        # force_M, force_Minv, force_f, force_xddot, force_alpha, force_frac, force_lag_M, force_lag_f= self._forced_geometry.evaluate(**kwargs)
        # print("M1", force_M)
        # print("f1", force_f)
        # print("xddot1", force_xddot)
        # print("alpha1", force_alpha)
        # print("frac1", force_frac)
        # print("lag_M1", lag_M)
        # print("lag_f1", lag_f)
        print("M1", M)
        print("f1", f)
        print("xddot1", xddot)
        print("alpha1", alpha)
        print("frac1", frac)
        print("lag_M1", lag_M)
        print("lag_f1", lag_f)
        # print("l_subst1", l)
        # print("S", S)
        # print("Mcond", Mcond)
        # print("M_subst1", M)
        # print("f_subst1", f)
        attractor = self.leaves['goal_0_leaf']
        attractor._map.concretize()
        x, xdot, J, Jdot = attractor._map.forward(**kwargs)
        Jt = np.transpose(J)
        not_pull_attractor = WeightedGeometry(g=attractor._geo, le= attractor._lag)
        not_pull_attractor.computeAlpha()
        attractor_M = ca.substitute(not_pull_attractor.M(), attractor._geo.x(), attractor._map._phi)
        attractor_M = ca.substitute(attractor_M, attractor._geo.xdot(), attractor._map._phidot)
        attractor_alpha = ca.substitute(not_pull_attractor._alpha, attractor._geo.x(), attractor._map._phi)
        attractor_alpha = ca.substitute(attractor_alpha, attractor._geo.xdot(), attractor._map._phidot)

        attractor_f = ca.substitute(not_pull_attractor.f(), attractor._geo.x(), attractor._map._phi)
        attractor_f = ca.substitute(attractor_f, attractor._geo.xdot(), attractor._map._phidot)
        attractor_frac = ca.substitute(not_pull_attractor.frac, attractor._geo.x(), attractor._map._phi)
        attractor_frac = ca.substitute(attractor_frac, attractor._geo.xdot(), attractor._map._phidot)
        attractor_xddot = ca.substitute(not_pull_attractor._xddot, attractor._geo.x(), attractor._map._phi)
        attractor_xddot = ca.substitute(attractor_xddot, attractor._geo.xdot(), attractor._map._phidot)
        attractor_le_M = ca.substitute(not_pull_attractor._le._S._M, attractor._geo.x(), attractor._map._phi)
        attractor_le_M = ca.substitute(attractor_le_M, attractor._geo.xdot(), attractor._map._phidot)
        attractor_le_f = ca.substitute(not_pull_attractor._le._S.f(), attractor._geo.x(), attractor._map._phi)
        attractor_le_f = ca.substitute(attractor_le_f, attractor._geo.xdot(), attractor._map._phidot)

        func = CasadiFunctionWrapper(
                "asdf", self.variables, {"attractor_M": attractor_M, "attractor_f": attractor_f, "attractor_frac" : attractor_frac,
                                        "attractor_alpha": attractor_alpha,
                                         "attractor_xddot": attractor_xddot, "attractor_le_M": attractor_le_M, "attractor_le_f": attractor_le_f}
            )
        evaluations = func.evaluate(**kwargs)
        attractor_M = evaluations["attractor_M"]
        attractor_f = evaluations["attractor_f"]
        attractor_alpha= evaluations["attractor_alpha"]
        attractor_frac = evaluations["attractor_frac"]
        attractor_xddot = evaluations["attractor_xddot"]
        attractor_le_M= evaluations["attractor_le_M"]
        attractor_le_f = evaluations["attractor_le_f"]
        # print("M1", evaluations["attractor_M"])
        # print("f1", evaluations["attractor_f"])
        # print("alpha1", evaluations["attractor_alpha"])
        # print("frac1", evaluations["attractor_frac"])
        # print("xddot1", evaluations["attractor_xddot"])
        # print("le_M1", evaluations["attractor_le_M"])
        # print("le_f1", evaluations["attractor_le_f"])
        # print("forced xddot", forced_xddot)

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
        forced_xddot = evaluations["forced_xddot"]
        # print("forced_xddot1", forced_xddot)
        # Debugging
        # a_ex = evaluations['a_ex']
        # beta = evaluations['beta']
        # exe_alpha = evaluations['exe_alpha']
        # force_alpha = evaluations['force_alpha']
        # logging.debug(f"a_ex: {evaluations['a_ex']}")
        # logging.debug(f"alhpa_forced_geometry: {evaluations['alpha_forced_geometry']}")
        # logging.debug(f"alpha_geometry: {evaluations['alpha_geometry']}")
        # logging.debug(f"beta : {evaluations['beta']}")
        action_magnitude = np.linalg.norm(action)
        if action_magnitude < eps:
            # logging.warning(f"Fabrics: Avoiding small action with magnitude {action_magnitude}")
            action *= 0.0
        elif action_magnitude > 1/eps:
            logging.warning(f"Fabrics: Avoiding large action with magnitude {action_magnitude}")
            action *= 0.0
        return action


