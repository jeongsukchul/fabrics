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
    # attractor_potential: str = (
    #     "5.0* (x/ca.norm_2(x) + 0.1 * (-2*10*x/ca.norm_2(x)/(1+ ca.exp(-2*10*ca.norm_2(x)))"
    # )
    attractor_metric: str = (
        "((2.0 - 0.3) * ca.exp(-(0.75 * ca.norm_2(x))**2) + 0.3) * ca.SX(np.identity(x.size()[0]))"
    )
    # attractor_metric: str = (
    #     " (0.75 * ca.norm_2(x))**2 * ca.SX(np.identity(x.size()[0]))"
    # )
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
EnergyExpression = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GeometryExpression = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
AttractorPotentialExpression = Callable[[torch.Tensor],torch.Tensor]
AttractorMetricExpression = Callable[[torch.Tensor],torch.Tensor]
DamperBetaExpression = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
DamperEtaExpression = Callable[[torch.Tensor], torch.Tensor]
Metric = Callable[[torch.Tensor, torch.Tensor],torch.Tensor]
Force = Callable[[torch.Tensor, torch.Tensor],torch.Tensor]
Geom = Callable[[torch.Tensor, torch.Tensor],torch.Tensor]
@dataclass
class TorchConfig:
    forcing_type: str = 'speed-controlled' # options are 'speed-controlled', 'pure-geometry', 'execution-energy', 'forced', 'forced-energized'
    norm_2 = lambda x: torch.sqrt(torch.sum(x**2,dim=-1))
    norm_2_2 = lambda x: torch.sum(x*82,dim=-1)
    base_energy: EnergyExpression = lambda x,xdot: (0.5*0.2 *torch.sum(xdot**2,dim=-1)).squeeze()
    base_M : Metric = lambda x, xdot: 0.2 * torch.eye(x.shape[-1]) 
    base_f : Force = lambda x, xdot: 0.2 * torch.zeros(x.shape[-1])
    collision_geometry: GeometryExpression = lambda x, xdot: -0.5 /(x**5) * (-0.5* torch.sign(xdot)-1)*xdot **2
    collision_finsler : EnergyExpression = lambda x, xdot: 0.1/(x**1) * xdot**2
    # collision_M: Metric = lambda x, xdot: torch.diag(0.2/x**1)
    # collision_f: Force : lambda x, xdot: - 0.2/(x**2) * xdot**2 +0.1/x**2 *xdot**2
    limit_geometry : GeometryExpression = lambda x, xdot: -0.1/x * xdot ** 2
    limit_finsler : EnergyExpression = lambda x, xdot: 0.1/x *(-0.5 * (torch.sign(xdot)-1)) * xdot**2
    limit_M : Metric = lambda x, xdot: torch.diag(0.2/x*(-0.5 * (torch.sign(xdot)-1)))
    limit_f : Force = lambda x, xdot: -0.2/(x**2)*(-0.5 *(torch.sign(xdot)-1))*xdot**2 + 0.1/(x**2) * (-0.5*(torch.sign(xdot)-1))*xdot**2
    self_collision_geometry : GeometryExpression = lambda x, xdot: -0.5/x *(-0.5*(torch.sign(xdot)-1))*xdot**2
    self_collision_finsler : EnergyExpression = lambda x, xdot: 0.1/x * xdot**2
    geometry_plane_constraint : GeometryExpression = lambda x, xdot: -0.5/(x**5) *(-0.5*(torch.sign(xdot)-1))*xdot**2
    finsler_plane_constraint : EnergyExpression = lambda x, xdot: 0.1/x *xdot**2
    attractor_metric : AttractorMetricExpression = lambda x, xdot: ((2.0-0.3)* torch.exp(-1*(0.75**2 * torch.sum(x**2,dim=-1)))+0.3)*torch.eye(x.shape[-1])
    attractor_force : Force = lambda x,xdot: (2.0-0.3)* torch.exp(-(0.75**2)*torch.sum(x**2,dim=-1))*(-2*(0.75**2)*x@xdot)*xdot\
        -0.5*((2.0-0.3)*torch.exp(-(0.75**2) * torch.sum(x**2,dim=-1)))*torch.sum(xdot**2,dim=-1)*(-2*(0.75**2)*x)
    
    # attractor_l:EnergyExpression = lambda x, xdot: 0.5*((2.0-0.3)* torch.exp(-1*(0.75**2 * torch.sum(x**2,dim=-1)))+0.3)*torch.sum(xdot**2,dim=-1)
    
    # attractor_dldx: Force = lambda x, xdot:0.5*((2.0-0.3)*torch.exp(-(0.75**2) * torch.sum(x**2,dim=-1)))*torch.sum(xdot**2,dim=-1)*(-2*(0.75**2)*x)
    # attractor_dldxdxdot: Metric = lambda x, xdot: (2.0-0.3)* torch.exp(-(0.75**2)*torch.sum(x**2,dim=-1))*(-2*(0.75**2)*x.unsqueeze(-1)@xdot.unsqueeze(-2))
    attractor_potential : AttractorPotentialExpression = lambda x, xdot:  5.0 * ( torch.sqrt(torch.sum(x ** 2,dim=-1)) + 0.1* torch.log(1+ torch.exp(-2 *10 * torch.sqrt(torch.sum(x ** 2, dim=-1))))) 
    attractor_geom : Geom = lambda x, xdot: 5.0 * (x/torch.sqrt(torch.sum(x**2,dim=-1)) + 0.1 * (-2*10*x)*torch.exp(-2 *10 * torch.sqrt(torch.sum(x ** 2, dim=-1)))/torch.sqrt(torch.sum(x**2,dim=-1)*(1+ torch.exp(-2*10*torch.sqrt(torch.sum(x**2,dim=-1))))))
    damper_beta : DamperBetaExpression = lambda x, a_ex, a_le: 0.5*(torch.tanh(-0.5* (torch.sqrt(torch.sum(x**2,dim=-1))-0.02))+1) * 6.5 + 0.01 + torch.maximum(torch.zeros_like(a_ex),a_ex-a_le)
    # damper_beta: str = (
    #     "0.5 * (ca.tanh(-0.5 * (ca.norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + ca.fmax(0, sym('a_ex') - sym('a_le'))"
    # )
    damper_eta : DamperEtaExpression = lambda xdot: 0.5 * (torch.tanh(-0.9* (1- 1/2)* torch.sum(xdot**2, dim=-1)-0.5)+1)
    
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
