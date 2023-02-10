import gym
import numpy as np
import casadi as ca
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from fabrics.components.energies.execution_energies import ExecutionLagrangian
from fabrics.diffGeometry.diffMap import ExplicitDifferentialMap
from fabrics.components.leaves.geometry import GenericGeometryLeaf
from fabrics.helpers.variables import Variables

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

class EDFGeometryLeaf(GenericGeometryLeaf):
    def __init__(
        self,
        parent_variables: Variables,
    ):
        phi = ca.SX.sym("edf_phi", 1)
        super().__init__(
            parent_variables,
            f"edf_leaf",
            phi,
        )
        self.set_forward_map()

    def set_forward_map(self):
        J = ca.transpose(ca.SX.sym("J_edf", 2))
        Jdot = ca.transpose(ca.SX.sym("Jdot_edf", 2))
        explicit_jacobians = {
            "phi_edf": self._forward_kinematics,
            "J_edf": J,
            "Jdot_edf": Jdot,
        }
        self._parent_variables.add_parameters(explicit_jacobians)
        self._forward_map =  ExplicitDifferentialMap(
            self._forward_kinematics,
            self._parent_variables,
            J=J,
            Jdot=Jdot,
        )

    def map(self):
        return self._forward_map

class EDFPlanner(ParameterizedFabricPlanner):
    def set_components(
        self,
        collision_links: list,
        goal: GoalComposition = None,
    ):
        geometry = EDFGeometryLeaf(self._variables)
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)
        if goal:
            self.set_goal_component(goal)
            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()

def edf(pos) -> float:
    #SARAYS FUNCTION
    # I am adding a random line :)
    kkk=1
    return 5.0

def edf_jacobian(pos) -> float:
    #SARAYS FUNCTION
    return np.zeros(2)

def initalize_environment(render):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
    
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "radius"])
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, 0.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 0,
                "child_link" : 1,
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_goal(goal.sub_goals()[0])
    env.add_obstacle(obst1)
    return (env, goal)

def set_planner(goal: GoalComposition):
    """
    Initializes the fabric planner for the point robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    degrees_of_freedom = 2
    robot_type = "pointRobot"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = EDFPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = [1]
    planner.set_components(
        collision_links,
        goal,
    )
    planner.concretize()
    return planner


def run_point_robot_urdf(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    The initial zero action step is needed to initialize the sensor in the
    urdf environment.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        action[0:2] = planner.compute_action(
            q=ob_robot["joint_state"]["position"][0:2],
            qdot=ob_robot["joint_state"]["velocity"][0:2],
            x_goal_0=ob_robot['FullSensor']['goals'][0][0][0:2],
            weight_goal_0=goal.sub_goals()[0].weight(),
            #x_obst_0=ob_robot['FullSensor']['obstacles'][0][0][0:2],
            #radius_obst_0=ob_robot['FullSensor']['obstacles'][0][1],
            phi_edf=edf(ob_robot["joint_state"]["position"][0:2]),
            J_edf=edf_jacobian(ob_robot["joint_state"]["position"][0:2]),
            Jdot_edf=edf_jacobian(ob_robot["joint_state"]["position"][0:2]),
            radius_body_1=np.array([0.2])
        )
        ob, *_, = env.step(action)
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=10000, render=True)
