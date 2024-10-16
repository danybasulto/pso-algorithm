import numpy as np

class Particle:
    """Represents a particle in the Particle Swarm Optimization (PSO).
    
    Examples
    --------
    >>> p1 = Particle(dimensions=5)
    
    >>> p2 = Particle(dimensions=10)
    
    >>> p3 = Particle(dimensions=20)
    
    Attributes:
        position (list[float]): The current position of the particle in the search space.
        fitness (float): The value of the target function at the current position.
        velocity (list[float]): The current velocity of the particle.
        best_position (list[float]): The best known position of the particle.
        best_fitness (float): The best value of the objective function found by the particle.
    """
    # Constructor
    def __init__(self, dimensions):
        """Initializes a new particle with random position and velocity.
        
        Args:
            dimensions (int): The number of dimensions of the search space.
        """
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("Las dimensiones deben ser un entero positivo.")
        
        self.__position = np.random.uniform(-5.12, 5.12, dimensions).tolist()
        self.__fitness = float('inf')
        self.__velocity = np.random.uniform(-1, 1, dimensions).tolist()
        self.__best_position = self.__position[:]
        self.__best_fitness = float('inf')
    # -- Getters & Setters --
    @property
    def position(self):
        return self.__position
    
    @position.setter
    def position(self, value):
        self.__position = value
    
    @property
    def fitness(self):
        return self.__fitness
    
    @fitness.setter
    def fitness(self, value):
        self.__fitness = value
    
    @property
    def velocity(self):
        return self.__velocity
    
    @velocity.setter
    def velocity(self, value):
        self.__velocity = value
    
    @property
    def best_position(self):
        return self.__best_position
    
    @best_position.setter
    def best_position(self, value):
        self.__best_position = value
    
    @property
    def best_fitness(self):
        return self.__best_fitness
    
    @best_fitness.setter
    def best_fitness(self, value):
        self.__best_fitness = value
# (c) 2024 Daniel Basulto del Toro & Juan Antonio Ramirez Aguilar
