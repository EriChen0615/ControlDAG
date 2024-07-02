from easydict import EasyDict
import functools
import logging

logger = logging.getLogger(__name__)


FeatureLoader_Registry = EasyDict() # registry for feature loaders
DataTransform_Registry = EasyDict() # registry for feature loaders
Model_Registry = EasyDict()
Executor_Registry = EasyDict()

def register_to(registry, name=None):
    def _register_func(func):
        register_func_to_registry(func, registry, name=name)
        @functools.wraps(func)
        def _func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return _func_wrapper
    return _register_func


def register_func_to_registry(func, registry, name=None):
    fn = name or func.__name__
    # assert fn not in registry, f"Cannot register {fn} due to duplicated name"
    if fn in registry:
        logger.warning(f"Overwriting {fn} in registry")
    registry[fn] = func

def register_model(cls):
    register_func_to_registry(cls, Model_Registry)
    return cls

def register_executor(cls):
    register_func_to_registry(cls, Executor_Registry)
    return cls

