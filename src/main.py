import numpy as np
from particle_swarm_optimization import ParticleSwarmOptimization

def run_experiment(dimensions, cognitive_coeff, iterations):
    """Run a PSO experiment and display the results.
    
    Args:
        dimensions (int): Number of dimensions for the PSO.
        cognitive_coeff (tuple): Cognitive coefficients (c1, c2) to use.
        iterations (int): Number of iterations of the algorithm.
        
    Examples
    --------
    >>> run_experiment(5, (1.0, 1.5))
    Best position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Best fitness: 0.0
    """
    print(f"\nRunning experiment with {dimensions} dimensions and coefficients: {cognitive_coeff}")
    
    # Initialize the PSO
    pso = ParticleSwarmOptimization(num_particles=30, dimensions=dimensions, cognitive_coeff=cognitive_coeff)
    
    # Run the optimization
    pso.optimize(iterations)
    
    # Get the best position and fitness
    best_position = pso.global_best_position
    best_fitness = pso.global_best_fitness
    
    # Show results
    print("Best position: ", best_position)
    print("Best fitness: ", best_fitness)

def main():
    """Main function to run multiple experiments of the Particle Swarm Optimization (PSO) algorithm.
    
    Examples
    --------
    >>> main()
    Running experiment with 5 dimensions...
    Best position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Best fitness: 0.0
    """
    # Experimental parameters
    iterations = 100
    cognitive_coefficients_list = [
        (1.0, 1.5),
        (1.5, 1.5),
        (2.0, 2.0)
    ]
    
    # Run experiments for 5, 10 and 20 dimensions
    for dimensions in [5, 10, 20]:
        for cognitive_coefficients in cognitive_coefficients_list:
            run_experiment(dimensions, cognitive_coefficients, iterations)
    
if __name__ == '__main__':
    main()
