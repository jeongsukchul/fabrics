import sys
from dataclasses import dataclass
from typing import List
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.components.robot_representation import RobotRepresentation
from fabrics.components.environment import Environment
import torch

@dataclass
class FabricPlannerConfig:
    forcing_type: str = 'speed-controlled' # options are 'speed-controlled', 'pure-geometry', 'execution-energy', 'forced', 'forced-energized'
    base_energy: str = (
        "0.5 * 0.2 * ca.dot(xdot, xdot)"
    )
    collision_geometry: str = (
        "-0.5 / (x ** 5) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    collision_finsler: str = (
        "0.1/(x**1) * xdot**2"
    )
    limit_geometry: str = (
        "-0.1 / (x ** 1) * xdot ** 2"
    )
    limit_finsler: str = (
        "0.1/(x**1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
    )
    self_collision_geometry: str = (
        "-0.5 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    self_collision_finsler: str = (
        "0.1/(x**1) * xdot**2"
    )
    geometry_plane_constraint: str = (
        "-0.5 / (x ** 5) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    finsler_plane_constraint: str = (
        "0.1/(x**1) * xdot**2"
    )
    attractor_potential: str = (
        "5.0 * (ca.norm_2(x) + 1 / 10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
    )
    attractor_metric: str = (
        "((2.0 - 0.3) * ca.exp(-1 * (0.75 * ca.norm_2(x))**2) + 0.3) * ca.SX(np.identity(x.size()[0]))"
    )
    damper_beta: str = (
        "0.5 * (ca.tanh(-0.5 * (ca.norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + ca.fmax(0, sym('a_ex') - sym('a_le'))"
    )
    damper_eta: str = (
        "0.5 * (ca.tanh(-0.9 * (1 - 1/2) * ca.dot(xdot, xdot) - 0.5) + 1)"
    )
    """
    damper_beta: str = (
        "0.5 * (ca.tanh(-sym('alpha_b') * (ca.norm_2(x) - sym('radius_shift'))) + 1) * sym('beta_close') + sym('beta_distant') + ca.fmax(0, sym('a_ex') - sym('a_le'))"
    )
    damper_eta: str = (
        "0.5 * (ca.tanh(-sym('alpha_eta') * sym('ex_lag') * (1 - sym('ex_factor')) - 0.5) + 1)"
    )
    """

from typing import Callable, Tuple
import torch
Energy = Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor]
Geometry = Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor],torch.Tensor,torch.Tensor]
attractorPotential = Tuple[Callable[[torch.Tensor],torch.Tensor],torch.Tensor]
attractorMetric = Tuple[Callable[[torch.Tensor],torch.Tensor],torch.Tensor]
damperBeta = Tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],torch.Tensor,torch.Tensor,torch.Tensor]
damperEta = Tuple[Callable[[torch.Tensor], torch.Tensor],torch.Tensor]
class TorchConfig:
    forcing_type: str = 'speed-controlled' # options are 'speed-controlled', 'pure-geometry', 'execution-energy', 'forced', 'forced-energized'
    def base_energy(xdot: torch.Tensor):
        return 0.5 * 0.2 * xdot**2 #"0.5 * 0.2 * ca.dot(xdot, xdot)"
    def collision_geometry(x: torch.Tensor, xdot: torch.Tensor):
        return -0.5 /(x**5) * (-0.5* torch.sign(xdot)-1)*xdot **2 #"-0.5 / (x ** 5) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    def collision_finsler(x:torch.Tensor, xdot: torch.Tensor): # "0.1/(x**1) * xdot**2"
        return 0.1/(x**1) * xdot**2
    def limit_geometry(x:torch.Tensor, xdot: torch.Tensor): #"-0.1 / (x ** 1) * xdot ** 2"
        return -0.1/x * xdot ** 2
    def limit_finsler(x:torch.Tensor, xdot: torch.Tensor): # 0.1/x * (-0.5 *  (ca.sign(xdot) - 1)) * xdot**2"
        return 0.1/x *(-0.5 * torch.sign(xdot)-1) * xdot**2
    def self_collision_geometry(x:torch.Tensor, xdot:torch.Tensor): #-0.5/x *(-0.5*(ca.sign(xdot) - 1)) * xdot ** 2"
        return -0.5/x *(-0.5*(torch.sign(xdot)-1))*xdot**2
    def self_collision_finsler(x:torch.Tensor, xdot:torch.Tensor):  #"0.1/(x**1) * xdot**2"
        return 0.1/x * xdot**2
    def geometry_plane_constraint(x:torch.Tensor, xdot:torch.Tensor): #"-0.5 / (x ** 5) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
        return -0.5/(x**5) *(-0.5*(torch.sign(xdot)-1))*xdot**2
    def finsler_plane_constraint(x:torch.Tensor, xdot:torch.Tensor): #0.1/(x**1)*xdot**2
        return 0.1/x *xdot**2
    def attractor_potential(x:torch.Tensor): #"5.0 * (ca.norm_2(x) + 1 / 10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))" 
        return 5.0 * (torch.linalg.norm(x,2) + 0.1* torch.log(1+ torch.exp(-2 *10 * torch.linalg.norm(x,2))))
    def attractor_metric(x: torch.Tensor): # "((2.0 - 0.3) * ca.exp(-1 * (0.75 * ca.norm_2(x))**2) + 0.3) * ca.SX(np.identity(x.size()[0]))"
        return ((2.0-0.3)* torch.exp(-1*(0.75* torch.linalg.norm(x,2)**2)+0.3)*torch.eye(x.shape[-1]))
    def damper_beta(x: torch.Tensor, a_ex: torch.Tensor, a_le: torch.Tensor): # "0.5 * (ca.tanh(-0.5 * (ca.norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + ca.fmax(0, sym('a_ex') - sym('a_le'))"
        return 0.5*(torch.tanh(-0.5* (torch.linalg.norm(x,2)-0.02))+1) * 6.5 + 0.01 + torch.max(0,a_ex-a_le)
    def damper_eta(xdot: torch.Tensor): # 0.5 * (ca.tanh(-0.9 * (1 - 1/2) * ca.dot(xdot, xdot) - 0.5) + 1)
        return 0.5 * (torch.tanh(-0.9* (1- 1/2)* torch.sum(xdot*xdot, dim=-1)-0.5)+1)
    
    def setBaseEnergy(self, x:torch.Tensor, xdot:torch.Tensor)->Energy:
        return (self.base_energy, xdot)
    def setCollisionGeometry(self,x:torch.Tensor, xdot:torch.Tensor)->Geometry:
        return (self.collision_geometry, x, xdot)
    def setCollisionFinsler(self,x:torch.Tensor, xdot:torch.Tensor)->Energy:
        return (self.collision_finsler, x, xdot)
    def setLimitGeometry(self,x:torch.Tensor, xdot:torch.Tensor)->Geometry:
        return (self.limit_geometry, x, xdot)
    def setLimitFinsler(self,x:torch.Tensor, xdot:torch.Tensor)->Energy:
        return (self.limit_finsler, x, xdot)
    def setGeometryPlaneConstraint(self,x:torch.Tensor, xdot:torch.Tensor)->Geometry:
        return (self.geometry_plane_constraint, x, xdot)
    def setFinslerPlaneConstraint(self,x:torch.Tensor, xdot:torch.Tensor)->Energy:
        return (self.finsler_plane_constraint, x, xdot)
    def setAttractorPotential(self,x:torch.Tensor, xdot:torch.Tensor)->attractorPotential:
        return (self.collision_geometry, x, xdot)
    def setAttractorMetric(self,x:torch.Tensor, xdot:torch.Tensor)->attractorMetric:
        return (self.collision_geometry, x, xdot)
    def setDamperBeta(self,x:torch.Tensor, a_ex:torch.Tensor, a_le:torch.Tensor)->damperBeta:
        return (self.collision_geometry, x, a_ex, a_le)
    def setDamperEta(self,xdot:torch.Tensor)->damperEta:
        return (self.collision_geometry, xdot)
@dataclass
class Subgoal:
    child_link: str
    desired_position: List[float]
    epsilon: float
    indices: List[int]
    is_primary_goal: bool
    parent_link: str
    type: str
    weight: float

@dataclass
class JointLimits:
    lower_limits: List[float]
    upper_limits: List[float]

class ProblemConfiguration:
    def __init__(self, **config):
        self._config = config
        self._goal_composition=GoalComposition(name='goal', content_dict=self._config['goal']['goal_definition'])
        self._joint_limits=JointLimits(
            lower_limits=self._config['joint_limits']['lower_limits'],
            upper_limits=self._config['joint_limits']['upper_limits']
        )
        self.construct_robot_representation()
        self._environment=Environment(
            number_spheres=self._config['environment']['number_spheres'],
            number_planes=self._config['environment']['number_planes'],
            number_cuboids=self._config['environment']['number_cuboids'],
        )

    def construct_robot_representation(self):
        collision_links = {}
        for link, link_data in self._config['robot_representation']['collision_links'].items():
            collision_link_type = list(link_data.keys())[0]
            collision_links[link] = getattr(sys.modules['fabrics.helpers.geometric_primitives'], collision_link_type.capitalize())(link, **link_data[collision_link_type])
        self._robot_representation=RobotRepresentation(
            collision_links=collision_links,
            self_collision_pairs=self._config['robot_representation']['self_collision_pairs']
        )

    @property
    def goal_composition(self) -> GoalComposition:
        return self._goal_composition

    @property
    def joint_limits(self) -> JointLimits:
        return self._joint_limits

    @property
    def robot_representation(self) -> RobotRepresentation:
        return self._robot_representation

    @property
    def environment(self) -> Environment:
        return self._environment
