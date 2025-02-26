import torch
import builtins
from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set

def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')

def _get_qualified_name(func: Callable[..., Any]) -> str:
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    return f'{module}.{name}'

def typename(target: Any) -> str:
    if isinstance(target, torch.nn.Module):
        return torch.typename(target)

    if isinstance(target, str):
        return target

    return _get_qualified_name(target)