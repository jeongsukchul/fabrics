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
from fabrics.components.energies.execution_energies import ExecutionLagrangian
from fabrics.components.leaves.attractor import GenericAttractor
from fabrics.components.leaves.dynamic_attractor import GenericDynamicAttractor
from fabrics.components.leaves.dynamic_geometry import (
    DynamicObstacleLeaf, GenericDynamicGeometryLeaf)
from fabrics.components.leaves.geometry import (AvoidanceLeaf,
                                                CapsuleCuboidLeaf,
                                                CapsuleSphereLeaf,
                                                ESDFGeometryLeaf,
                                                GenericGeometryLeaf, LimitLeaf,
                                                ObstacleLeaf,
                                                PlaneConstraintGeometryLeaf,
                                                SelfCollisionLeaf,
                                                SphereCuboidLeaf)
from fabrics.components.leaves.leaf import Leaf
from fabrics.diffGeometry.diffMap import (DifferentialMap,
                                          DynamicDifferentialMap)
from fabrics.diffGeometry.energized_geometry import WeightedGeometry, TorchWeightedGeometry
from fabrics.diffGeometry.energy import Lagrangian, TorchLagrangian
from fabrics.diffGeometry.geometry import Geometry, TorchGeometry
from fabrics.diffGeometry.speedControl import Damper
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper
from fabrics.helpers.constants import eps
from fabrics.helpers.exceptions import ExpressionSparseError
from fabrics.helpers.functions import is_sparse, parse_symbolic_input
from fabrics.helpers.geometric_primitives import Sphere
from fabrics.helpers.variables import Variables, TorchVariables
from fabrics.planner.configuration_classes import (FabricPlannerConfig, TorchConfig,
                                                   ProblemConfiguration)

import torch

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
        # self._target_velocity = np.zeros(self._geometry.x().size()[0]) #cart dof
        self._ref_sign = 1
        self.leaves = {}

    """ INIT"""

    def initialize_joint_varibles(self):
        q = torch.zeros(self._dof)
        qdot = torch.zeros(self._dof)
        self._variables = TorchVariables(state_variables={"q": q, "qdot": qdot})

    def set_base_geometry(self):
        q = self._variables.position_variable()
        qdot = self._variables.velocity_variable() 
        base_energy = self._config.base_energy
        base_geometry = TorchGeometry(h = torch.zeros(self._dof), var=self._variables)
        base_lagrangian = TorchLagrangian(base_energy, var=self._variables)
        self._geometry = TorchWeightedGeometry(g=base_geometry, le=base_lagrangian)
    
    """ ADDING COMPONENTS"""
    def add_geometry(
        self, forward_map: DifferentialMap, lagrangian: TorchLagrangian, geometry: Geometry
    ) -> None:
        weighted_geometry = WeightedGeometry(g=geometry, le=lagrangian)
        self.add_weighted_geometry(forward_map, weighted_geometry)
    
    def add_weighted_geometry(
        self, forward_map: DifferentialMap, weighted_geometry: TorchWeightedGeometry
    ) -> None:
        pulled_geometry = weighted_geometry.pull(forward_map)
        self._geometry += pulled_geometry
        self._variables = self._variables + pulled_geometry._vars
    
    def add_forcing_geometry(
        self,
        forward_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
        prime_forcing_leaf: bool,
    ) -> None:
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
        
    def add_leaf(self, leaf: Leaf, prime_leaf: bool= False) -> None:
        print("leaf, ", leaf._leaf_name)
        if isinstance(leaf, TorchGenericAttractor):
            self.add_forcing_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry(), prime_leaf)
        elif isinstance(leaf, TorchGenericGeometryLeaf):
            self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())
        self.leaves[leaf._leaf_name] = leaf

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
        print("-----------variables after init", self._variables)

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
        print("leaf, ", leaf._leaf_name)
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

    def set_joint_limits(self):
        limits = np.zeros((self._dof, 2))
        limits[:, 0] = self._problem_configuration.joint_limits.lower_limits
        limits[:, 1] = self._problem_configuration.joint_limits.upper_limits
        limits = limits.tolist()
        for joint_index in range(len(limits)):
            self.add_limit_geometry(joint_index, limits[joint_index])

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
        print("variables", self._variables)
        if goal:
            
            self.set_goal_component(goal)
            print("---------------variables after set_goal", self._variables)

            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            print("----------------variables after set_execution energy", self._variables)

            # Sets speed control
            self.set_speed_control()
            print("---------------variables after set_speed_control", self._variables)


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
            if sub_goal.type() in ["analyticSubGoal", "splineSubGoal"]:
                attractor = GenericDynamicAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            else:
                self._variables.add_parameter(f'x_goal_{j}', ca.SX.sym(f'x_goal_{j}', sub_goal.dimension()))
                attractor = GenericAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            attractor.set_potential(self.config.attractor_potential)
            attractor.set_metric(self.config.attractor_metric)
            self.add_leaf(attractor, prime_leaf=sub_goal.is_primary_goal())


    def concretize(self, mode='acc', time_step=None):
        print("planner concretized with mode :", mode)
        print("config_ forcing type ", self._config.forcing_type)
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
        evaluations = self._funs.evaluate(**kwargs)
        action = evaluations["action"]
        # Debugging
        #logging.debug(f"a_ex: {evaluations['a_ex']}")
        #logging.debug(f"alhpa_forced_geometry: {evaluations['alpha_forced_geometry']}")
        #logging.debug(f"alpha_geometry: {evaluations['alpha_geometry']}")
        #logging.debug(f"beta : {evaluations['beta']}")
        action_magnitude = np.linalg.norm(action)
        if action_magnitude < eps:
            logging.warning(f"Fabrics: Avoiding small action with magnitude {action_magnitude}")
            action *= 0.0
        elif action_magnitude > 1/eps:
            logging.warning(f"Fabrics: Avoiding large action with magnitude {action_magnitude}")
            action *= 0.0
        return action


