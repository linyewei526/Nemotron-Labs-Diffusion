#!/usr/bin/env python3
"""
DLM Generation Registry

This module provides a registry system for different DLM generation algorithms.
It allows for easy registration and discovery of generation methods.
"""

import logging
from typing import Dict, List, Optional, Type
from .base import GenerationAlgorithm

logger = logging.getLogger(__name__)


class GenerationRegistry:
    """Registry for DLM generation algorithms."""
    
    def __init__(self):
        self._algorithms: Dict[str, GenerationAlgorithm] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(self, algorithm: GenerationAlgorithm, aliases: Optional[List[str]] = None):
        """
        Register a generation algorithm.
        
        Args:
            algorithm: The algorithm instance to register
            aliases: Optional list of alternative names for the algorithm
        """
        if not isinstance(algorithm, GenerationAlgorithm):
            raise TypeError("Algorithm must be an instance of GenerationAlgorithm")
        
        name = algorithm.name
        if name in self._algorithms:
            logger.warning(f"Algorithm '{name}' is already registered. Overwriting.")
        
        self._algorithms[name] = algorithm
        logger.info(f"Registered generation algorithm: {name}")
        
        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    logger.warning(f"Alias '{alias}' is already registered. Overwriting.")
                self._aliases[alias] = name
                logger.debug(f"Registered alias '{alias}' -> '{name}'")
    
    def get(self, name: str) -> Optional[GenerationAlgorithm]:
        """Get a generation algorithm by name or alias."""
        # Check if it's a direct name
        if name in self._algorithms:
            return self._algorithms[name]
        
        # Check if it's an alias
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._algorithms.get(actual_name)
        
        return None
    
    def list_algorithms(self) -> List[str]:
        """List all registered algorithm names."""
        return list(self._algorithms.keys())
    
    def list_available_algorithms(self) -> List[str]:
        """List all registered algorithms that are currently available."""
        available = []
        for name, algorithm in self._algorithms.items():
            if algorithm.is_available():
                available.append(name)
        return available
    
    def get_algorithm_info(self, name: str) -> Optional[Dict[str, str]]:
        """Get information about an algorithm."""
        algorithm = self.get(name)
        if algorithm is None:
            return None
        
        return {
            'name': algorithm.name,
            'description': algorithm.description,
            'engine': algorithm.engine,
            'available': algorithm.is_available()
        }
    
    def list_algorithms_by_engine(self, engine: str) -> List[str]:
        """List all registered algorithms for a specific engine."""
        return [name for name, algo in self._algorithms.items() if algo.engine == engine]
    
    def list_available_algorithms_by_engine(self, engine: str) -> List[str]:
        """List available algorithms for a specific engine."""
        return [name for name, algo in self._algorithms.items() 
                if algo.engine == engine and algo.is_available()]
    
    def get_default_algorithm_for_engine(self, engine: str) -> Optional[str]:
        """Get a recommended default algorithm for an engine."""
        engine_defaults = {
            'nemotron': 'nemotron',
            'ar_native': 'ar_native',
        }
        default_name = engine_defaults.get(engine)
        if default_name and self.get(default_name):
            return default_name
        
        # Fallback: return first available algorithm for this engine
        available = self.list_available_algorithms_by_engine(engine)
        return available[0] if available else None
    
    def clear(self):
        """Clear all registered algorithms."""
        self._algorithms.clear()
        self._aliases.clear()


# Global registry instance
registry = GenerationRegistry()


def register_algorithm(algorithm: GenerationAlgorithm, aliases: Optional[List[str]] = None):
    """Register a generation algorithm in the global registry."""
    registry.register(algorithm, aliases)


def get_algorithm(name: str) -> Optional[GenerationAlgorithm]:
    """Get a generation algorithm from the global registry."""
    return registry.get(name)


def list_algorithms() -> List[str]:
    """List all registered algorithm names."""
    return registry.list_algorithms()


def list_available_algorithms() -> List[str]:
    """List all available algorithm names."""
    return registry.list_available_algorithms()


def get_algorithm_info(name: str) -> Optional[Dict[str, str]]:
    """Get information about an algorithm."""
    return registry.get_algorithm_info(name)


def list_algorithms_by_engine(engine: str) -> List[str]:
    """List all algorithms for a specific engine."""
    return registry.list_algorithms_by_engine(engine)


def list_available_algorithms_by_engine(engine: str) -> List[str]:
    """List available algorithms for a specific engine."""
    return registry.list_available_algorithms_by_engine(engine)


def get_default_algorithm_for_engine(engine: str) -> Optional[str]:
    """Get the recommended default algorithm for an engine."""
    return registry.get_default_algorithm_for_engine(engine)


# Built-in algorithm registration. Only the four diffusion-family decoders this
# slim build serves are registered. (fast_dllm/dinfer/dllm_eval/huggingface
# were upstream LLaDA-family helpers; their source packages were deleted.)
def _register_builtin_algorithms():
    from .nemotron import NemotronGeneration
    register_algorithm(NemotronGeneration(), aliases=['nemotron', 'nemotron_native', 'nemotron_diffusion'])

    from .nemotron_mixed import NemotronMixedGeneration
    register_algorithm(NemotronMixedGeneration(), aliases=['nemotron_mixed', 'mix_ar_dlm'])

    from .ar_native import ArNativeGeneration
    register_algorithm(ArNativeGeneration(), aliases=['ar_native', 'ar-native'])

_register_builtin_algorithms()




# Export main components
__all__ = [
    'GenerationRegistry',
    'GenerationAlgorithm',
    'registry',
    'register_algorithm',
    'get_algorithm',
    'list_algorithms',
    'list_available_algorithms',
    'get_algorithm_info',
    'list_algorithms_by_engine',
    'list_available_algorithms_by_engine',
    'get_default_algorithm_for_engine',
]
