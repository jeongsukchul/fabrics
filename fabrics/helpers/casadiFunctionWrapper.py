from copy import copy
import pdb
import casadi as ca
from fabrics.helpers.variables import Variables, TorchVariables
import numpy as np
import os
import pickle
import _pickle as cPickle
import bz2
import logging
import torch, functorch
from fabrics.helpers.constants import eps

# from functorch._C import GradTrackingTensor

class InputMissmatchError(Exception):
    pass

class TorchFunctionWrapper(object):
    _instances = []
    grad_count = 0
    def __init__(self, variables: TorchVariables, expression=None, function=None, ex_input=None, name = None, iscasadi= False, use_cache=True):
        
        self._variables = variables
        self._inputs = variables.asArray()
        self._name = name
        self._use_cache = use_cache
        self._cache = None
        if expression is not None:
            self._expression = expression 
            self._ex_input = ex_input
            def func(**kwargs):
                # print("ex_input", self._ex_input)
                # print("kwargs", [kwargs[input] for input in self._ex_input])
                return self._expression(*[kwargs[input] for input in self._ex_input]).type(torch.float64)
            self._func = func
            # self._func = lambda **kwargs : self._expression(*{kwargs[input] for input in ex_input})
        else:
            self._func = function
        TorchFunctionWrapper._instances.append(self)
    def TorchFunctionOperator(self, other,operator_type='add'):
        combined_variables = self._variables + other._variables
        # print(self._name , " ",  operator_type ," ", other._name)

        def combined_func(**kwargs):
            result1 = self(**{input: kwargs[input] for input in self._inputs})
            result2 = other(**{input: kwargs[input] for input in other._inputs})
                
            # print("f1 name", self._name)
            # print("f2 name", other._name)
            # print("opeartor type", operator_type)

            # print("f1 input",{input: kwargs[input] for input in self._inputs})
            # print("f2 input",{input: kwargs[input] for input in other._inputs})

            # print("result1 ", result1)
            # print("result2", result2)

            if operator_type == 'add':
                # print("operate result", result1 + result2)
                return result1 + result2
            if operator_type == 'sub':
                # print("operate result", result1 - result2)
                return result1 - result2
            if operator_type == 'matmul':
                # print("operate result", result1 @ result2)
                return result1 @ result2
            if operator_type == 'mul':
                # print("operate result", result1 * result2)
                return result1 * result2
            if operator_type == 'div':
                # print("operate result", result1 / result2)
                if result2.item() == 0:
                    raise Exception("divide by zero")
                return result1 / result2
            if operator_type == 'dot':
                return torch.sum(result1 * result2, dim=-1)
                # return torch.tensordot(result1, result2, dims=([-1], [-1]))
        return TorchFunctionWrapper(function=combined_func, variables=combined_variables)
    def __neg__(self):
        func = lambda **kwargs : -self(**kwargs)
        return TorchFunctionWrapper(function=func, variables=self._variables)
    def __mul__(self, other):
        if isinstance(other, TorchFunctionWrapper):
            return self.TorchFunctionOperator(other, 'mul')
        else:
            func = lambda **kwargs : self(**kwargs)*other
            return TorchFunctionWrapper(function=func, variables=self._variables)
    def __rmul__(self,other):
        func = lambda **kwargs : other*self(**kwargs)
        return TorchFunctionWrapper(function=func, variables=self._variables)
    def __truediv__(self, other):
        if isinstance(other, TorchFunctionWrapper):
            return self.TorchFunctionOperator(other, 'div')
        else:
            func = lambda **kwargs : self(**kwargs)/other
            return TorchFunctionWrapper(function=func, variables=self._variables)
    def __rtruediv__(self,other):
        func = lambda **kwargs : other/self(**kwargs)
        return TorchFunctionWrapper(function=func, variables=self._variables)

    def __add__(self, other):
        if isinstance(other, TorchFunctionWrapper):
            return self.TorchFunctionOperator(other, 'add')
        else: 
            func = lambda **kwargs : self(**kwargs)+other
            return TorchFunctionWrapper(function=func, variables=self._variables)

    def __radd__(self, other):
        func = lambda **kwargs : self(**kwargs)+other            
        return TorchFunctionWrapper(function=func, variables=self._variables)
    def __sub__(self,other):        
        if isinstance(other, TorchFunctionWrapper):
            return self.TorchFunctionOperator(other, 'sub')
        else:
            func = lambda **kwargs : self(**kwargs)-other
            return TorchFunctionWrapper(function=func, variables=self._variables)
    def __rsub__(self,other):
        func = lambda **kwargs : other - self(**kwargs)            
        return TorchFunctionWrapper(function=func, variables=self._variables)
    def __matmul__(self, other):
        if isinstance(other, TorchFunctionWrapper):
            return self.TorchFunctionOperator(other, 'matmul')
        elif isinstance(other, torch.Tensor):
            # def matmulfunc(**kwargs):
            #     func = self._func(**kwargs)
            #     print("func", func)
            #     print("other", other)
            #     return func@ other
            func = lambda **kwargs : self(**kwargs) @ other            
            return TorchFunctionWrapper(function= func, variables=self._variables) 
        else:
            raise TypeError   
    def __rmatmul__(self, other):
        if isinstance(other, torch.Tensor):
            func = lambda **kwargs : other @ self(**kwargs)          
            return TorchFunctionWrapper(function=func, variables=self._variables) 
        else:
            raise TypeError   
    def sum(self):
        def sum_fn(**kwargs):
            return torch.sum(self(**kwargs),dim=-1)
        return TorchFunctionWrapper(function=sum_fn, variables=self._variables)
    def diag(self):
        def diag_fn(**kwargs):
            M = self(**kwargs)
            M_diag = torch.diag(M)
            return M_diag
            # func = lambda **kwargs: torch.linalg.pinv(self._func(**kwargs)+ torch.eye(s.size()[0]) * eps))
        return TorchFunctionWrapper(function=diag_fn, variables=self._variables)
    def pinv(self):
        def inv(**kwargs):
            M = self(**kwargs)
            M_reg = M+ eps*torch.eye(M.size(0))

            # U, S, V = torch.svd(M_reg)
            # # print("S", S)
            # Mcond = torch.linalg.cond(M_reg)
            # print("Mcond", Mcond)
            M_inv =  torch.linalg.pinv(M_reg).type(torch.float64)
            # print("Minv",M_inv)
            return M_inv
            # func = lambda **kwargs: torch.linalg.pinv(self._func(**kwargs)+ torch.eye(s.size()[0]) * eps))
        return TorchFunctionWrapper(function=inv, variables=self._variables)
    def dot(self, other):
        return self.TorchFunctionOperator(other,'dot')
    def transpose(self):

        func= lambda **kwargs: torch.transpose(self(**kwargs),-2,-1)
        return TorchFunctionWrapper(function=func, variables=self._variables)
    

    def lowerLeaf(self, dm):
        from fabrics.diffGeometry.diffMap import TorchDifferentialMap
        assert isinstance(dm, TorchDifferentialMap)
        q = dm._vars._position
        qdot = dm._vars._velocity
        x = self._variables._position
        xdot = self._variables._velocity
        vars = TorchVariables(position=q, velocity =qdot,\
                              parameter_variables= (dm._vars._parameter_variables | self._variables._parameter_variables))
        def lowerLeafFunc(**lower_leaf_kwargs):
            # print("lower_leaf kwargs", lower_leaf_kwargs)
            
            upper_leaf_kwargs = copy(lower_leaf_kwargs)
            upper_leaf_kwargs[x] = dm._phi(**lower_leaf_kwargs)
            upper_leaf_kwargs[xdot] = dm._J(**lower_leaf_kwargs) @ lower_leaf_kwargs[qdot]
            del upper_leaf_kwargs[q]
            del upper_leaf_kwargs[qdot]
            
            # print("J", dm._J(**lower_leaf_kwargs))

            # print("upper+leaf_kwargs", upper_leaf_kwargs)
            return self(**upper_leaf_kwargs)
        return TorchFunctionWrapper(function=lowerLeafFunc, variables = vars)

    def set_name(self, name:str):
        self._name = name
    def name(self):
        return self._name

    def grad(self, variable, end_grad=False):
        if variable not in self._inputs:
            raise Exception("Gradient variable is not in the function!")

        index = self._inputs.index(variable)
        def wrapper_fn(*args):
            # print("grad called by", self._name)
            # print("grad variable", variable)
            kwargs = {self._inputs[i]: args[i] for i in range(len(args))}
            # result = self(**kwargs)
            # print("grad function output", result)

            return self._func(**kwargs)
        grad_fn = torch.func.jacrev(wrapper_fn, argnums=index)
        def grad_func(**kwargs):
            TorchFunctionWrapper.grad_count+=1

            args = [kwargs[input] for input in self._inputs]
            result = grad_fn(*args)
            if end_grad:
                return result.detach().type(torch.float64)
            else:
                return result.type(torch.float64)
        return TorchFunctionWrapper(function=grad_func, variables=self._variables)
    
    def grad_elementwise(self, variable, end_grad=False):

        if variable not in self._inputs:
            raise Exception("Gradient variable is not in the function!")

        index = self._inputs.index(variable)
    
        def wrapper_fn(*args):
            # print("grad element called by", self._name)
            kwargs = {self._inputs[i]: args[i] for i in range(len(args))}
            # print("grad element function output", self(**kwargs))

            return self._func(**kwargs)

        grad_fn = torch.func.jacrev(wrapper_fn, argnums=index)

        def grad_func_elementwise(**kwargs): 
            TorchFunctionWrapper.grad_count+=1
            args = [kwargs[input] for input in self._inputs]
            if end_grad:
                # print("grad result",torch.diag(grad_fn(*args),0).detach().type(torch.float64))
                return torch.diag(grad_fn(*args),0).detach().type(torch.float64)
            else:
                # print("grad result", torch.diag(grad_fn(*args),0).type(torch.float64))
                return torch.diag(grad_fn(*args),0).type(torch.float64)
        
        return TorchFunctionWrapper(function=grad_func_elementwise, variables=self._variables)
    def hessian(self, variable1, variable2, end_grad=False):
        if variable1 not in self._inputs or variable2 not in self._inputs:
            raise Exception("Gradient variable is not in the function!")

        index1 = self._inputs.index(variable1)
        index2 = self._inputs.index(variable2)
        def wrapper_fn(*args):
            kwargs = {self._inputs[i]: args[i] for i in range(len(args))}
            return self._func(**kwargs)
        hessian_fn = torch.func.jacrev(torch.func.jacrev(wrapper_fn, argnums=index1), argnums=index2)
        def hessian_func(**kwargs):
            args = [kwargs[input] for input in self._inputs]
            if end_grad:
                return hessian_fn(*args).detach().type(torch.float64)
            else:
                return hessian_fn(*args).type(torch.float64)
        return TorchFunctionWrapper(function=hessian_func, variables=self._variables)
    
    def __call__(self, **kwargs):
        if self._cache is None:
            if self._use_cache:
                self._cache = self._func(**kwargs)
                return self._cache
            else:
                return self._func(**kwargs)
        else:
            # print("use_cache", self._cache)
            return self._cache
    @classmethod
    def reset_all_caches(cls):
        for instance in cls._instances:
            instance.reset_cache()

    def reset_cache(self):
        self._cache = None
class CasadiFunctionWrapper(object):

    def __init__(self, name: str, variables: Variables, expressions: dict):
        self._name = name
        self._inputs = variables.asDict()
        self._expressions = expressions
        self._argument_dictionary = variables.parameters_values()
        self.create_function()

    def create_function(self):
        self._input_keys = sorted(tuple(self._inputs.keys()))
        self._input_sizes = {i: self._inputs[i].size() for i in self._inputs}
        self._list_expressions = [self._expressions[i] for i in sorted(self._expressions.keys())]
        input_expressions = [self._inputs[i] for i in self._input_keys]
        self._function = ca.Function(self._name, input_expressions, self._list_expressions)

    def function(self) -> ca.Function:
        return self._function

    def serialize(self, file_name):
        with bz2.BZ2File(file_name, 'w') as f:
            pickle.dump(self._function.serialize(), f)
            pickle.dump(list(self._expressions.keys()), f)
            pickle.dump(self._input_keys, f)
            pickle.dump(self._argument_dictionary, f)
    def detach_numpy(self,tensor):
        tensor = tensor.detach().cpu()
        if torch._C._functorch.is_gradtrackingtensor(tensor):
            tensor = torch._C._functorch.get_unwrapped(tensor)
            return np.array(tensor.storage().tolist()).reshape(tensor.shape)
        return tensor.numpy()
    def evaluate(self, **kwargs):
        # print("evaluate : ", kwargs)
        for key in kwargs: # pragma no cover
            # print("key:",key)
            if isinstance(kwargs[key],torch.Tensor):
                # value = torch.tensor(kwargs[key].cpu().numpy())
                kwargs[key] = self.detach_numpy(kwargs[key])
            if key == 'x_obst' or key == 'x_obsts':
                obstacle_dictionary = {}
                for j, x_obst_j in enumerate(kwargs[key]):
                    obstacle_dictionary[f'x_obst_{j}'] = x_obst_j
                self._argument_dictionary.update(obstacle_dictionary)
            if key == 'radius_obst' or key == 'radius_obsts':
                radius_dictionary = {}
                for j, radius_obst_j in enumerate(kwargs[key]):
                    radius_dictionary[f'radius_obst_{j}'] = radius_obst_j
                self._argument_dictionary.update(radius_dictionary)
            if key == 'x_obst_dynamic' or key == 'x_obsts_dynamic':
                obstacle_dyn_dictionary = {}
                for j, x_obst_dyn_j in enumerate(kwargs[key]):
                    obstacle_dyn_dictionary[f'x_obst_dynamic_{j}'] = x_obst_dyn_j
                self._argument_dictionary.update(obstacle_dyn_dictionary)
            if key == 'xdot_obst_dynamic' or key == 'xdot_obsts_dynamic':
                xdot_dyn_dictionary = {}
                for j, xdot_obst_dyn_j in enumerate(kwargs[key]):
                    xdot_dyn_dictionary[f'xdot_obst_dynamic_{j}'] = xdot_obst_dyn_j
                self._argument_dictionary.update(xdot_dyn_dictionary)
            if key == 'xddot_obst_dynamic' or key == 'xddot_obsts_dynamic':
                xddot_dyn_dictionary = {}
                for j, xddot_obst_dyn_j in enumerate(kwargs[key]):
                    xddot_dyn_dictionary[f'xddot_obst_dynamic_{j}'] = xddot_obst_dyn_j
                self._argument_dictionary.update(xddot_dyn_dictionary)
            if key == 'radius_obst_dynamic' or key == 'radius_obsts_dynamic':
                radius_dyn_dictionary = {}
                for j, radius_obst_dyn_j in enumerate(kwargs[key]):
                    radius_dyn_dictionary[f'radius_obst_dynamic_{j}'] = radius_obst_dyn_j
                self._argument_dictionary.update(radius_dyn_dictionary)
            if key == 'x_obst_cuboid' or key == 'x_obsts_cuboid':
                x_obst_cuboid_dictionary = {}
                for j, x_obst_cuboid_j in enumerate(kwargs[key]):
                    x_obst_cuboid_dictionary[f'x_obst_cuboid_{j}'] = x_obst_cuboid_j
                self._argument_dictionary.update(x_obst_cuboid_dictionary)
            if key == 'size_obst_cuboid' or key == 'size_obsts_cuboid':
                size_obst_cuboid_dictionary = {}
                for j, size_obst_cuboid_j in enumerate(kwargs[key]):
                    size_obst_cuboid_dictionary[f'size_obst_cuboid_{j}'] = size_obst_cuboid_j
                self._argument_dictionary.update(size_obst_cuboid_dictionary)
            if key.startswith('radius_body') and key.endswith('links'):
                # Radius bodies can be passed using a dictionary where the keys are simple integers.
                radius_body_dictionary = {}
                body_size_inputs = [input_exp for input_exp in self._input_keys if input_exp.startswith('radius_body')]
                for link_nr, radius_body_j in kwargs[key].items():
                    try:
                        key = [body_size_input for body_size_input in body_size_inputs if str(link_nr) in body_size_input][0]
                    except IndexError as e:
                        logging.warning(f"No body link with index {link_nr} in the inputs. Body link {link_nr} is ignored.")
                    radius_body_dictionary[key] = radius_body_j
                self._argument_dictionary.update(radius_body_dictionary)
            else:
                self._argument_dictionary[key] = kwargs[key]
        input_arrays = []
        # print("constraint_0 : ", kwargs["constraint_0"])
        try:
            input_arrays = [self._argument_dictionary[i] for i in self._input_keys]
        except KeyError as e:
            msg = f"Key {e} is not contained in the inputs\n"
            msg += f"Possible keys are {self._input_keys}\n"
            msg += f"You provided {list(kwargs.keys())}\n"
            raise InputMissmatchError(msg)
        try:
            list_array_outputs = self._function(*input_arrays)
        except RuntimeError as runtime_error:
            raise InputMissmatchError(runtime_error.args)
        output_dict = {}
        if isinstance(list_array_outputs, ca.DM):
            return {list(self._expressions.keys())[0]: np.array(list_array_outputs)[:, 0]}
        for i, key in enumerate(sorted(self._expressions.keys())):
            raw_output = list_array_outputs[i]
            if raw_output.size() == (1, 1):
                output_dict[key] = np.array(raw_output)[:, 0]
            elif raw_output.size()[1] == 1:
                output_dict[key] = np.array(raw_output)[:, 0]
            else:
                output_dict[key] = np.array(raw_output)
        return output_dict

    # def casadi_to_torchfunction(self):
    #     def wrapped_func(**kwargs):
    #         args = [kwargs[arg] for arg in self._input_keys]
    #         return self._function(*args) # Convert to scalar if single output
    #     inputs = self._input.keys()
    #     position = inputs[0]
    #     velocity = inputs[1]
    #     torchvariables = TorchVariables(position = position, velocity = velocity, parameter_variables=set(inputs[2:]))
    #     return TorchFunctionWrapper(wrapped_func, torchvariables)
class CasadiFunctionWrapper_deserialized(CasadiFunctionWrapper):

    def __init__(self, file_name: str):
        if os.path.isfile(file_name):
            logging.info(f"Initializing casadiFunctionWrapper from {file_name}")
            data = bz2.BZ2File(file_name, 'rb')
            self._function = ca.Function().deserialize(cPickle.load(data))
            expression_keys = cPickle.load(data)
            self._input_keys = cPickle.load(data)
            self._argument_dictionary = cPickle.load(data)
            self._expressions = {}
            for key in expression_keys:
                self._expressions[key] = []
            self._isload = True


