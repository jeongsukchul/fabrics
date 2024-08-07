import gymnasium as gym
import os
import numpy as np
import quaternionic

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner, TorchPlanner

import time
import torch
# TODO: Angle cannot be read through the FullSensor

def initalize_environment(render=True, obstacle_resolution = 8):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="panda.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    ).unwrapped
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
    )
    # q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
    q0 = np.array([0.0, -1.0, 0.0, -2.5, 0.0, 0, 0.0])
    # Definition of the obstacle.
    radius_ring = 0.3
    obstacles = []
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
    # goal_orientation = [1.0, 0.0, 0.0, 0.0]
    rotation_matrix = quaternionic.array(goal_orientation).to_rotation_matrix
    whole_position = [-0, 0, 2] #[0.1, 0.6, 0.8] # [0.3, 0.2, 1]
    # whole_position_obstacles = [10.1, 10.6, 10.8]
    # for i in range(obstacle_resolution + 1):
    #     angle = i/obstacle_resolution * 2.*np.pi
    #     origin_position = [
    #         0.0,
    #         radius_ring * np.cos(angle),
    #         radius_ring * np.sin(angle),
    #     ]
    #     position = np.dot(np.transpose(rotation_matrix), origin_position) + whole_position_obstacles
    #     static_obst_dict = {
    #         "type": "sphere",
    #         "geometry": {"position": position.tolist(), "radius": 0.1},
    #     }
    #     obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        # "subgoal1": {
        #     "weight": 1,
        #     "is_primary_goal": True,
        #     "indices": [0, 1, 2],
        #     "parent_link": "panda_link7",
        #     "child_link": "panda_hand",
        #     "desired_position": [0.107, 0.0, 0.0],
        #     "angle": goal_orientation,
        #     "epsilon": 0.05,
        #     "type": "staticSubGoal",
        # }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=q0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7, obstacle_resolution = 10):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """

    robot_type = 'panda'

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "5.0 * (ca.norm_2(x) + 1 /10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
    # damper = {
    #     "alpha_b": 0.5,
    #     "alpha_eta": 0.5,
    #     "alpha_shift": 0.5,
    #     "beta_distant": 0.01,
    #     "beta_close": 6.5,
    #     "radius_shift": 0.1,
    # }
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     robot_type,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    # attractor_potential = "15.0 * (ca.norm_2(x) + 1 /10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
    # collision_geometry= "-0.1 / (x ** 2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    # collision_finsler= "0.1/(x**1) * xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert_polluted_2.urdf", "r", encoding='utf-8') as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="panda_link0",
        end_links=["panda_vacuum", "panda_vacuum_2"],
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    planner2 = TorchPlanner(degrees_of_freedom, forward_kinematics)
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]    
    panda_limits = [
            [1-1.8973, 2.8973-1],
            [1-1.7628, 1.7628-1],
            [1-2.8973, 2.8973-1],
            [1-3.0718, -0.0698-1],
            [1-2.8973, 2.8973-1],
            [1-0.0175, 3.7525-1],
            [1-2.8973, 2.8973-1]
        ]
    panda_limits_torch = torch.tensor(panda_limits, dtype=torch.float64)
    
    collision_links = ['panda_link1', 'panda_link4', 'panda_link6', 'panda_hand']
    self_collision_pairs = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=obstacle_resolution,
        limits=panda_limits,
    )
    planner2.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=obstacle_resolution,
        limits=panda_limits_torch,
    )
    planner.concretize()
    return planner, planner2


def run_panda_ring_example(n_steps=5000, render=True, serialize=False, planner=None):
    torch.set_printoptions(precision=8)
    obstacle_resolution_ring = 0
    (env, goal) = initalize_environment(
        render=render,
        obstacle_resolution=obstacle_resolution_ring
    )
    action = np.zeros(7)
    ob, *_ = env.step(action)
    env.reconfigure_camera(1.4000000953674316, 67.9999008178711, -31.0001220703125, (-0.4589785635471344, 0.23635289072990417, 0.3541859984397888))
    if not planner:
        planner1, planner2 = set_planner(goal, obstacle_resolution = obstacle_resolution_ring)
        # Serializing the planner is optional
        if serialize:
            planner.serialize('serialized_10.pbz2')

    # sub_goal_0_quaternion = quaternionic.array(goal.sub_goals()[1].angle())
    goal_orientation = [1.0, 0.0, 0.0, 0.0]
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]

    sub_goal_0_quaternion = quaternionic.array(goal_orientation)
    
    sub_goal_0_rotation_matrix = sub_goal_0_quaternion.to_rotation_matrix
    obstacle_resolution_ring-=1
    cur_time = time.time()
    print("=======================================================================================")
    ob_robot = ob['robot_0']
    x_obsts = [
        ob_robot['FullSensor']['obstacles'][i+2]['position'] for i in range(obstacle_resolution_ring)
    ]
    radius_obsts = [
        ob_robot['FullSensor']['obstacles'][i+2]['size'] for i in range(obstacle_resolution_ring)
    ]

    # planner1.test_init_function(
    #     q=ob_robot["joint_state"]["position"],
    #     qdot=ob_robot["joint_state"]["velocity"],
    #     x_obsts=x_obsts,
    #     radius_obsts=radius_obsts,
    #     x_goal_0=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['position'],
    #     weight_goal_0=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['weight'],
    #     x_goal_1=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['position'],
    #     weight_goal_1=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['weight'],
    #     radius_body_panda_link1=0.1,
    #     radius_body_panda_link4=0.1,
    #     radius_body_panda_link6=0.15,
    #     radius_body_panda_hand=0.1,
    #     angle_goal_1=np.array(sub_goal_0_rotation_matrix),
    # )
    
    # planner2.test_init_function(
    #     q=torch.from_numpy(ob_robot["joint_state"]["position"]).type(torch.float64),
    #     qdot=torch.from_numpy(ob_robot["joint_state"]["velocity"]).type(torch.float64),
    #     x_obsts=x_obsts,
    #     radius_obsts=radius_obsts,
    #     x_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['position']).type(torch.float64),
    #     weight_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['weight']).type(torch.float64),
    #     x_goal_1=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['position']).type(torch.float64),
    #     weight_goal_1=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['weight']).type(torch.float64),
    #     radius_body_panda_link1=0.1,
    #     radius_body_panda_link4=0.1,
    #     radius_body_panda_link6=0.15,
    #     radius_body_panda_hand=0.1,
    #     angle_goal_1=torch.tensor(sub_goal_0_rotation_matrix, dtype= torch.float64),
    # )
    
    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        x_obsts = [
            ob_robot['FullSensor']['obstacles'][i+2]['position'] for i in range(obstacle_resolution_ring)
        ]
        radius_obsts = [
            ob_robot['FullSensor']['obstacles'][i+2]['size'] for i in range(obstacle_resolution_ring)
        ]
        prev_time = time.time()
        action1 = planner1.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_obsts=x_obsts,
            radius_obsts=radius_obsts,
            x_goal_0=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['weight'],
            # x_goal_1=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['position'],
            # weight_goal_1=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['weight'],
            radius_body_panda_link1=0.1,
            radius_body_panda_link4=0.1,
            radius_body_panda_link6=0.15,
            radius_body_panda_hand=0.1,
            # angle_goal_1=np.array(sub_goal_0_rotation_matrix),
        )
        panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
        panda_limits = [
            [1-1.8973, 2.8973-1],
            [1-1.7628, 1.7628-1],
            [1-2.8973, 2.8973-1],
            [1-3.0718, -0.0698-1],
            [1-2.8973, 2.8973-1],
            [1-0.0175, 3.7525-1],
            [1-2.8973, 2.8973-1]
        ]
        panda_limits = torch.tensor(panda_limits, dtype=torch.float64)
        cur_time = time.time()
        print("dt1", cur_time-prev_time)
        q=torch.from_numpy(ob_robot["joint_state"]["position"]).type(torch.float64)
        
        print("lower_limit", q-panda_limits[:,0])
        print("upper_limit", panda_limits[:,1]-q)
        prev_time = time.time()
        # qdot=torch.from_numpy(ob_robot["joint_state"]["velocity"]).type(torch.float64),
        # # x_obsts=x_obsts,
        # # radius_obsts=radius_obsts,
        # x_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['position']).type(torch.float64),
        # weight_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['weight']).type(torch.float64),
        # radius_body_panda_link1=0.1,
        # radius_body_panda_link4=0.1,
        # radius_body_panda_link6=0.15,
        # radius_body_panda_hand=0.1,
    
        # # Number of copies/batch size
        # batch_size = 5

        # # Create batched tensors by repeating and stacking
        # q_batched = torch.stack([q]*batch_size)
        # qdot_batched = torch.stack([qdot]*batch_size)
        # x_goal_0_batched = torch.stack([x_goal_0]*batch_size)
        # weight_goal_0_batched = torch.stack([weight_goal_0]*batch_size)

        # # Create batched radius values
        # radius_body_panda_link1_batched = torch.tensor([radius_body_panda_link1]*batch_size)
        # radius_body_panda_link4_batched = torch.tensor([radius_body_panda_link4]*batch_size)
        # radius_body_panda_link6_batched = torch.tensor([radius_body_panda_link6]*batch_size)
        # radius_body_panda_hand_batched = torch.tensor([radius_body_panda_hand]*batch_size)
        # batched_inputs = {
        #     'q': q_batched,
        #     'qdot': qdot_batched,
        #     'x_goal_0': x_goal_0_batched,
        #     'weight_goal_0': weight_goal_0_batched,
        #     'radius_body_panda_link1': radius_body_panda_link1_batched,
        #     'radius_body_panda_link4': radius_body_panda_link4_batched,
        #     'radius_body_panda_link6': radius_body_panda_link6_batched,
        #     'radius_body_panda_hand': radius_body_panda_hand_batched
        # }
        action2= planner2.compute_action(
            q=torch.from_numpy(ob_robot["joint_state"]["position"]).type(torch.float64),
            qdot=torch.from_numpy(ob_robot["joint_state"]["velocity"]).type(torch.float64),
            x_obsts=x_obsts,
            radius_obsts=radius_obsts,
            x_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['position']).type(torch.float64),
            weight_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['weight']).type(torch.float64),
            # x_goal_1=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['position']).type(torch.float64),
            # weight_goal_1=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['weight']).type(torch.float64),
            radius_body_panda_link1=0.1,
            radius_body_panda_link4=0.1,
            radius_body_panda_link6=0.15,
            radius_body_panda_hand=0.1,
            # angle_goal_1=torch.tensor(sub_goal_0_rotation_matrix, dtype= torch.float64),
        )
        
        cur_time = time.time()
        print("dt2", cur_time-prev_time)
        # U,S1,V = np.linalg.svd(M1)
        # U,S2,V = np.linalg.svd(M2)
        print("action1", action1)
        print("action2", action2)

        # print("S1", S1)
        # print("S2", S2)
        # print("J1", J1)
        # print("J2", J2)
        # print("J",np.sum(J2.numpy()-J1))
        # print("Jdot",np.sum(Jdot2.numpy()-Jdot1))
        # print("qdot",J2.numpy()@ -Jdot1))
        # ob, *_ = env.step(action2.numpy())
        ob, *_ = env.step(action1)
        

    # for _ in range(n_steps):
    #     ob_robot = ob['robot_0']
    #     x_obsts = [
    #         ob_robot['FullSensor']['obstacles'][i+2]['position'] for i in range(obstacle_resolution_ring)
    #     ]
    #     radius_obsts = [
    #         ob_robot['FullSensor']['obstacles'][i+2]['size'] for i in range(obstacle_resolution_ring)
    #     ]
    #     action = planner2.compute_action(
    #     q=torch.from_numpy(ob_robot["joint_state"]["position"]).type(torch.float64),
    #     qdot=torch.from_numpy(ob_robot["joint_state"]["velocity"]).type(torch.float64),
    #     x_obsts=x_obsts,
    #     radius_obsts=radius_obsts,
    #     x_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['position']).type(torch.float64),
    #     weight_goal_0=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['weight']).type(torch.float64),
    #     # x_goal_1=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['position']).type(torch.float64),
    #     # weight_goal_1=torch.from_numpy(ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['weight']).type(torch.float64),
    #     radius_body_panda_link1=0.1,
    #     radius_body_panda_link4=0.1,
    #     radius_body_panda_link6=0.15,
    #     radius_body_panda_hand=0.1,
    #     # angle_goal_1=torch.tensor(sub_goal_0_rotation_matrix, dtype= torch.float64),
    #     )
    #     output = action.detach().numpy()
    #     # print("action", type(output))
    #     ob, *_ = env.step(output)
    #     cur_time = time.time()
    #     print("dt", cur_time-prev_time)
    #     prev_time = cur_time
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_ring_example(n_steps=10000, serialize = False)
