"""
PromeTorch Neural Network Module
"""

from .._C.nn import (
    # C++ modules (used internally)
    Module as _CppModule,

    # Containers
    Sequential as _CppSequential,
    ModuleList as _CppModuleList,

    # Linear
    Linear as _CppLinear,

    # Activations
    ReLU as _CppReLU,
    ReLU6,
    LeakyReLU,
    PReLU,
    ELU,
    SELU,
    Sigmoid as _CppSigmoid,
    Tanh as _CppTanh,
    GELU as _CppGELU,
    SiLU as _CppSiLU,
    Mish,
    Softmax as _CppSoftmax,
    LogSoftmax,
    Softplus,
    Softsign,
    Hardtanh,
    Hardsigmoid,
    Hardswish,

    # Normalization
    BatchNorm2d,
    LayerNorm as _CppLayerNorm,

    # Dropout
    Dropout as _CppDropout,

    # Convolution
    Conv2d,

    # Pooling
    MaxPool2d,
    AvgPool2d,

    # Sparse
    Embedding as _CppEmbedding,

    # Recurrent
    RNN,
    LSTM,
    GRU,

    # Loss
    MSELoss,
    CrossEntropyLoss,
    NLLLoss,
    BCELoss,
    L1Loss,

    # Reduction enum
    Reduction,

    # Submodules
    functional,
    utils,

    # Init functions (exposed via nn.init below)
    zeros_ as _zeros,
    ones_ as _ones,
    uniform_ as _uniform,
    normal_ as _normal,
    xavier_uniform_ as _xavier_uniform,
    kaiming_uniform_ as _kaiming_uniform,
    orthogonal_ as _orthogonal,
)

# Alias for functional
import promethorch.nn.functional as F


# ============================================================================
# Pure-Python Module base class
# ============================================================================
# The C++ Module can't track Python-level submodules/parameters,
# so we provide a pure-Python Module that PIR and other Python models use.

class Module:
    """
    Pure-Python nn.Module compatible with PromeTorch tensors.
    Supports parameters(), named_parameters(), children(), train/eval, etc.
    """
    def __init__(self):
        self._parameters = {}    # name -> Parameter or Tensor
        self._modules = {}       # name -> Module
        self._buffers = {}       # name -> Tensor (non-parameter)
        self._training = True

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        # Check if it's a Parameter
        if isinstance(value, Parameter):
            self._parameters[name] = value
            object.__setattr__(self, name, value)
            return

        # Check if it's a Module (Python or C++)
        if isinstance(value, (Module, _CppModule)):
            self._modules[name] = value
            object.__setattr__(self, name, value)
            return

        # Check if it's a ModuleList
        if isinstance(value, ModuleList):
            self._modules[name] = value
            object.__setattr__(self, name, value)
            return

        # Regular attribute
        object.__setattr__(self, name, value)

    def register_buffer(self, name, tensor):
        """Register a buffer (non-parameter tensor)."""
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def parameters(self):
        """Return all parameters (recursively)."""
        params = []
        for name, p in self._parameters.items():
            if isinstance(p, Parameter):
                params.append(p.data)
            else:
                params.append(p)

        for name, m in self._modules.items():
            if isinstance(m, Module):
                params.extend(m.parameters())
            elif isinstance(m, ModuleList):
                for sub in m:
                    if isinstance(sub, Module):
                        params.extend(sub.parameters())
                    elif isinstance(sub, _CppModule):
                        for pp in sub.parameters():
                            params.append(pp)
            elif isinstance(m, _CppModule):
                for pp in m.parameters():
                    params.append(pp)
        return params

    def named_parameters(self, prefix=''):
        """Return all named parameters (recursively)."""
        result = []
        for name, p in self._parameters.items():
            full_name = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            if isinstance(p, Parameter):
                result.append((full_name, p.data))
            else:
                result.append((full_name, p))

        for name, m in self._modules.items():
            child_prefix = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            if isinstance(m, Module):
                result.extend(m.named_parameters(child_prefix))
            elif isinstance(m, ModuleList):
                for i, sub in enumerate(m):
                    sub_prefix = f"{child_prefix}.{i}"
                    if isinstance(sub, Module):
                        result.extend(sub.named_parameters(sub_prefix))
                    elif isinstance(sub, _CppModule):
                        for np_pair in sub.named_parameters():
                            result.append((f"{sub_prefix}.{np_pair[0]}", np_pair[1]))
            elif isinstance(m, _CppModule):
                for np_pair in m.named_parameters():
                    result.append((f"{child_prefix}.{np_pair[0]}", np_pair[1]))
        return result

    def children(self):
        """Return immediate child modules."""
        return list(self._modules.values())

    def named_children(self):
        """Return immediate named child modules."""
        return list(self._modules.items())

    def train(self, mode=True):
        """Set training mode."""
        self._training = mode
        for m in self._modules.values():
            if hasattr(m, 'train'):
                m.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def is_training(self):
        return self._training

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.parameters():
            if hasattr(p, 'grad') and p.grad is not None and p.grad.defined():
                p.grad.zero_()

    def to(self, device_or_dtype):
        """Move all parameters and buffers to device/dtype."""
        for name, p in self._parameters.items():
            if isinstance(p, Parameter):
                p._data = p._data.to(device_or_dtype)
            else:
                self._parameters[name] = p.to(device_or_dtype)
        for name, buf in self._buffers.items():
            self._buffers[name] = buf.to(device_or_dtype)
            object.__setattr__(self, name, self._buffers[name])
        for m in self._modules.values():
            if hasattr(m, 'to'):
                m.to(device_or_dtype)
        return self

    def state_dict(self):
        """Return state dict."""
        state = {}
        for name, p in self._parameters.items():
            if isinstance(p, Parameter):
                state[name] = p.data
            else:
                state[name] = p
        for name, buf in self._buffers.items():
            state[name] = buf
        for mod_name, m in self._modules.items():
            if hasattr(m, 'state_dict'):
                child_state = m.state_dict()
                for k, v in child_state.items():
                    state[f"{mod_name}.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, m in self._modules.items():
            lines.append(f"  ({name}): {repr(m)}")
        lines.append(")")
        return "\n".join(lines)


# ============================================================================
# nn.Parameter — wraps a Tensor to mark it as a learnable parameter
# ============================================================================

class Parameter:
    """A Tensor that is a module parameter (requires_grad=True by default)."""
    def __init__(self, data, requires_grad=True):
        if requires_grad:
            data.requires_grad_(True)
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __repr__(self):
        return f"Parameter containing:\n{self._data}"

    # Delegate attribute access to the underlying tensor
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._data, name)


# ============================================================================
# nn.Identity — pass-through module
# ============================================================================

class Identity(Module):
    """A module that returns its input unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# ============================================================================
# Python-wrapped C++ modules (to work with our Python Module)
# ============================================================================

class Linear(Module):
    """Linear layer wrapping the C++ implementation."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self._cpp = _CppLinear(in_features, out_features, bias)
        # Expose weight/bias as module attributes for nn.init access
        self._in_features = in_features
        self._out_features = out_features

    @property
    def weight(self):
        return self._cpp.weight

    @weight.setter
    def weight(self, value):
        # For weight tying: just store the reference
        self._tied_weight = value

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    def forward(self, x):
        return self._cpp.forward(x)

    def parameters(self):
        return list(self._cpp.parameters())

    def named_parameters(self, prefix=''):
        result = []
        for name, p in self._cpp.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            result.append((full_name, p))
        return result

    def to(self, device_or_dtype):
        self._cpp.to(device_or_dtype)
        return self

    def train(self, mode=True):
        self._cpp.train(mode)
        return self

    def eval(self):
        self._cpp.eval()
        return self

    def state_dict(self):
        return self._cpp.state_dict()

    def __repr__(self):
        return f"Linear(in_features={self._in_features}, out_features={self._out_features})"


class Embedding(Module):
    """Embedding layer wrapping the C++ implementation."""
    def __init__(self, num_embeddings, embedding_dim, padding_idx=-1, max_norm=0.0, sparse=False):
        super().__init__()
        self._cpp = _CppEmbedding(num_embeddings, embedding_dim, padding_idx, max_norm, sparse)

    @property
    def weight(self):
        return self._cpp.weight

    def forward(self, x):
        return self._cpp.forward(x)

    def parameters(self):
        return list(self._cpp.parameters())

    def named_parameters(self, prefix=''):
        result = []
        for name, p in self._cpp.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            result.append((full_name, p))
        return result

    def to(self, device_or_dtype):
        self._cpp.to(device_or_dtype)
        return self

    def state_dict(self):
        return self._cpp.state_dict()


class LayerNorm(Module):
    """LayerNorm wrapping C++ implementation."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self._cpp = _CppLayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        return self._cpp.forward(x)

    def parameters(self):
        return list(self._cpp.parameters())

    def named_parameters(self, prefix=''):
        result = []
        for name, p in self._cpp.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            result.append((full_name, p))
        return result

    def to(self, device_or_dtype):
        self._cpp.to(device_or_dtype)
        return self


class Dropout(Module):
    """Dropout wrapping C++ implementation."""
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self._cpp = _CppDropout(p, inplace)

    def forward(self, x):
        return self._cpp.forward(x)

    def parameters(self):
        return []

    def named_parameters(self, prefix=''):
        return []

    def train(self, mode=True):
        self._cpp.train(mode)
        return self

    def eval(self):
        self._cpp.eval()
        return self


class ModuleList(Module):
    """Python ModuleList that tracks child modules."""
    def __init__(self, modules=None):
        super().__init__()
        self._module_list = []
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module):
        idx = len(self._module_list)
        self._module_list.append(module)
        self._modules[str(idx)] = module

    def __len__(self):
        return len(self._module_list)

    def __getitem__(self, idx):
        return self._module_list[idx]

    def __iter__(self):
        return iter(self._module_list)

    def parameters(self):
        params = []
        for m in self._module_list:
            if hasattr(m, 'parameters'):
                params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=''):
        result = []
        for i, m in enumerate(self._module_list):
            child_prefix = f"{prefix}.{i}" if prefix else str(i)
            if hasattr(m, 'named_parameters'):
                result.extend(m.named_parameters(child_prefix))
        return result

    def to(self, device_or_dtype):
        for m in self._module_list:
            if hasattr(m, 'to'):
                m.to(device_or_dtype)
        return self

    def train(self, mode=True):
        for m in self._module_list:
            if hasattr(m, 'train'):
                m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        state = {}
        for i, m in enumerate(self._module_list):
            if hasattr(m, 'state_dict'):
                child_state = m.state_dict()
                for k, v in child_state.items():
                    state[f"{i}.{k}"] = v
        return state

    def __repr__(self):
        lines = ["ModuleList("]
        for i, m in enumerate(self._module_list):
            lines.append(f"  ({i}): {repr(m)}")
        lines.append(")")
        return "\n".join(lines)


# ============================================================================
# nn.init — initialization functions
# ============================================================================

class _InitModule:
    """nn.init namespace with initialization functions."""

    @staticmethod
    def orthogonal_(tensor, gain=1.0):
        """Orthogonal initialization."""
        return _orthogonal(tensor, gain)

    @staticmethod
    def normal_(tensor, mean=0.0, std=1.0):
        """Normal initialization."""
        return _normal(tensor, mean, std)

    @staticmethod
    def uniform_(tensor, a=0.0, b=1.0):
        """Uniform initialization."""
        return _uniform(tensor, a, b)

    @staticmethod
    def zeros_(tensor):
        """Fill with zeros."""
        return _zeros(tensor)

    @staticmethod
    def ones_(tensor):
        """Fill with ones."""
        return _ones(tensor)

    @staticmethod
    def xavier_uniform_(tensor, gain=1.0):
        """Xavier uniform initialization."""
        return _xavier_uniform(tensor, gain)

    @staticmethod
    def kaiming_uniform_(tensor, a=0.0, mode='fan_in', nonlinearity='leaky_relu'):
        """Kaiming uniform initialization."""
        return _kaiming_uniform(tensor, a, mode, nonlinearity)

init = _InitModule()


# Re-export activation modules as simple wrappers
class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self._cpp = _CppReLU(inplace)
    def forward(self, x):
        return self._cpp.forward(x)
    def parameters(self):
        return []
    def named_parameters(self, prefix=''):
        return []

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.sigmoid()
    def parameters(self):
        return []
    def named_parameters(self, prefix=''):
        return []

class Tanh(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.tanh()
    def parameters(self):
        return []
    def named_parameters(self, prefix=''):
        return []

class GELU(Module):
    def __init__(self, approximate='none'):
        super().__init__()
        self._cpp = _CppGELU(approximate)
    def forward(self, x):
        return self._cpp.forward(x)
    def parameters(self):
        return []
    def named_parameters(self, prefix=''):
        return []

class SiLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * x.sigmoid()
    def parameters(self):
        return []
    def named_parameters(self, prefix=''):
        return []

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self._dim = dim
        self._cpp = _CppSoftmax(dim)
    def forward(self, x):
        return self._cpp.forward(x)
    def parameters(self):
        return []
    def named_parameters(self, prefix=''):
        return []


# Export Sequential that works with our Python Module
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m
        self._module_list = list(modules)

    def forward(self, x):
        for m in self._module_list:
            x = m(x)
        return x

    def parameters(self):
        params = []
        for m in self._module_list:
            if hasattr(m, 'parameters'):
                params.extend(m.parameters())
        return params


__all__ = [
    'Module',
    'Sequential',
    'ModuleList',
    'Linear',
    'ReLU',
    'ReLU6',
    'LeakyReLU',
    'PReLU',
    'ELU',
    'SELU',
    'Sigmoid',
    'Tanh',
    'GELU',
    'SiLU',
    'Mish',
    'Softmax',
    'LogSoftmax',
    'Softplus',
    'Softsign',
    'Hardtanh',
    'Hardsigmoid',
    'Hardswish',
    'BatchNorm2d',
    'LayerNorm',
    'Dropout',
    'Conv2d',
    'MaxPool2d',
    'AvgPool2d',
    'Embedding',
    'RNN',
    'LSTM',
    'GRU',
    'MSELoss',
    'CrossEntropyLoss',
    'NLLLoss',
    'BCELoss',
    'L1Loss',
    'Reduction',
    'functional',
    'utils',
    'F',
    'Parameter',
    'Identity',
    'init',
]
