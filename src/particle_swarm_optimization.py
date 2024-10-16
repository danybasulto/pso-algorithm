import numpy as np
from particle import Particle

class ParticleSwarmOptimization:
    """Implementation of the Particle Swarm Optimization (PSO).
    
    Attributes:
        __num_particles (int): Number of the particles in the swarm.
        __dimensions (int): Dimensions of the search space.
        __particles (list[Particle]): List of the particles in the swarm.
        __global_best_position (list[float]): Best known position for the entire swarm.
        __global_best_fitness (float): Best value of the objective function known by the whole swarm.
        __cognitive_coeff (float): Cognitive coefficient.
        __social_coeff (float): Social coefficient.
        __inertia_weight (float): Inertia weight.
        
    Examples
    --------
    >>> 
    """
    def __init__(self, num_particles, dimensions, cognitive_coeff=1.5, social_coeff=1.5, inertia_weight=0.5):
        """Initializes the PSO with a set of particles.
        
        Args:
            num_particles (int): number of the particles in the swarm.
            dimensions (int): Dimensions of the search space.
            cognitive_coeff (float): Cognitive coefficient.
            social_coeff (float): Social coefficient.
            inertia_weight (float): Inertia weight.
        """
        self.__num_particles = num_particles
        self.__dimensions = dimensions
        self.__cognitive_coeff = cognitive_coeff
        self.__social_coeff = social_coeff
        self.__inertia_weight = inertia_weight
        
        # Initialize particles
        self.__particles = [Particle(dimensions) for _ in range(num_particles)]
        
        # Initialize best global position
        self.__global_best_position = None
        self.__global_best_fitness = float('inf')
        self.__initialize_global_best()
        
    def __initialize_global_best(self):
        """It establishes the best global position and value from the initial particles."""
        for particle in self.__particles:
            if particle.best_fitness < self.__global_best_fitness:
                self.__global_best_fitness = particle.best_fitness
                self.__global_best_position = particle.best_position
                    
    def __update_particles(self):
        """It updates the position and velocity of each particle in the swarm."""
        for particle in self.__particles:
            # Upgrade velocity
            new_velocity = (self.__inertia_weight * particle.velocity +
                            self.__cognitive_coeff * np.random.rand() * (particle.best_position - particle.position) +
                            self.__social_coeff * np.random.randn() * (self.__global_best_position - particle.position))
            particle.velocity = new_velocity.tolist()
            
            # Upgrade position
            particle.position = (particle.position + particle.velocity).tolist()
            
            # Evaluate new value
            particle.fitness = self.__evaluate_fitness(particle.position)
            
            # Upgrade better local position
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position
            
            # Upgrade better global position
            if particle.fitness < self.__global_best_fitness:
                self.__global_best_fitness = particle.fitness
                self.__global_best_position = particle.position
    
    def __evaluate_fitness(self, position):
        """Evaluates the objective function at the given position.
        
        Args:
            position (list[float]): The current position of the particle.
            
        Returns:
            float: Value of the target function.
        """
        return sum(100.0 * (position[i+1] - position[i]**2.0)**2.0 + (1 - position[i])**2.0 for i in range(len(position) - 1))
    
    def optimize(self, iterations):
        """Runs the PSO algorithm for a specified number of iterations.
        
        Args:
            iterations (int): Number of iterations to optimize.
        """
        for _ in range(iterations):
            self.__update_particles()
    
    @property
    def global_best_position(self):
        """Returns the best known global position.
        
        Returns:
            list[float]: Best global position.
        """
        return self.__global_best_position
    
    @property
    def global_best_fitness(self):
        """Returns the best known global value.
        
        Returns:
            float: Global best value.
        """
        return self.__global_best_fitness
# (c) 2024 Daniel Basulto del Toro & Juan Antonio Ramirez Aguilar
