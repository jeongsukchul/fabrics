import casadi as ca
from fabrics.diffGeometry.energy import Lagrangian, TorchLagrangian
import torch
from fabrics.helpers.casadiFunctionWrapper import TorchFunctionWrapper

class ExecutionLagrangian(Lagrangian):
    def __init__(self, var):
        xdot = var.velocity_variable()
        le = ca.dot(xdot, xdot)
        super().__init__(le, var=var)

class TorchExecutionLagrangian(TorchLagrangian):
    def __init__(self, var):
        x = var.position_variable()
        xdot = var.velocity_variable()
        le_ex = lambda x,xdot: torch.sum(xdot*xdot, dim=-1)
        le = TorchFunctionWrapper(expression=le_ex, variables=var, ex_input=[x,xdot], name="TorchExecutionLagrangian")
        super().__init__(le, var=var)

