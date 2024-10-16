import numpy as np

class Particle:
    """Represents a particle in the Particle Swarm Optimization (PSO).
    
    Examples
    --------
    >>> p1 = Particle(dimensions=5)
    
    >>> p2 = Particle(dimensions=10)
    
    >>> p3 = Particle(dimensions=20)
    
    Attributes:
        __position (np.ndarray): The current position of the particle in the search space.
        __fitness (float): The value of the target function at the current position.
        __velocity (np.ndarray): The current velocity of the particle.
        __best_position (np.ndarray): The best known position of the particle.
        __best_fitness (float): The best value of the objective function found by the particle.
    """
    
    def __init__(self, dimensions):
        """Initializes a new particle with random position and velocity.
        
        Args:
            dimensions (int): The number of dimensions of the search space.
        """
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("Las dimensiones deben ser un entero positivo.")
        
        self.__position = np.random.uniform(-5.12, 5.12, dimensions)
        self.__fitness = float('inf')
        self.__velocity = np.random.uniform(-1, 1, dimensions)
        self.__best_position = self.__position.copy()
        self.__best_fitness = float('inf')
    
    # -- Getters --
    @property
    def position(self):
        return self.__position
    
    @property
    def fitness(self):
        return self.__fitness
    
    @property
    def velocity(self):
        return self.__velocity
    
    @property
    def best_position(self):
        return self.__best_position
    
    @property
    def best_fitness(self):
        return self.__best_fitness
# (c) 2024 Daniel Basulto del Toro & Juan Antonio Ramirez Aguilar
