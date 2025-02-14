import os
import neat
import gymnasium as gym
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scale_image

scaling_mode = True
random = True
runs = 3
generations = 50

# Reduce input image resolution depending on number of inputs
def scale_observation(observation, inputs):
    scaled_observation = []
    input_count = len(inputs)

    # Cut image into parts
    image_parts = scale_image.deconstruct_image(observation, input_count)

    # Get average value for each part
    for i in range(len(image_parts)):
        image_parts[i] = scale_image.average_pixels(image_parts[i])
    
    # Assign to inputs (pick average pixel value of each image part)
    for i in range (1, len(observation.flatten())+1):
        if (not inputs or inputs[0]!= -i):
            scaled_observation.append(0)
        else:
            scaled_observation.append(image_parts[0][0][0]) # Append first pixel of first row of first image part
            inputs = inputs[1:]
            image_parts = image_parts[1:]
    return scaled_observation

# Do one run with the provided genome
def run_genome(genome, config, render_mode = None):

    # Create neural network from genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Initialize reward
    total_reward = 0

    # Run genome x times and get average fitness
    for r in range(runs):
        # Create environment
        env = gym.make("CarRacing-v2", lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode=render_mode)
        if (random):
            observation, info = env.reset()
        else:
            observation, info = env.reset(seed=42)

        # Convert observation from RGB to grayscale (gray = 0.2989 * r + 0.5870 * g + 0.1140 * b)
        observation = np.dot(observation[...,:3], [0.299, 0.587, 0.114])

        # Run either with scaling or plain NEAT
        if (scaling_mode == True):

            # Count number of connected inputs and get input IDs
            inputs = []
            for connection in genome.connections:
                if (connection[0] < 0 and connection[0] not in inputs):
                    inputs.append(connection[0])
            inputs.sort(reverse=True)

            # Scale observation according to number of inputs
            observation = scale_observation(observation, inputs)

        else:
            observation = observation.flatten()

        # Run simulation and calculate fitness
        for _ in range(500):
            action = net.activate(observation)
            action = action.index(max(action)) 
            observation, reward, terminated, truncated, info = env.step(action)
            observation = np.dot(observation[...,:3], [0.299, 0.587, 0.114])
            
            # Run either with scaling or plain NEAT
            if (scaling_mode == True):
                observation = scale_observation(observation, inputs)
            else:
                observation = observation.flatten()

            total_reward += reward
            if terminated or truncated:
                env.close()
                break
        
    return total_reward / runs

# Evaluate fitness of provided genomes
def eval_genomes(genomes, config, render_mode = 'none'):
    for genome_id, genome in genomes:
        genome.fitness = run_genome(genome, config)

# Run evolution
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to X generations.
    winner = p.run(eval_genomes, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Run simulation with winning genome
    run_genome(winner, config, "human")
    while (True):
        reply = input("Replay winning solution? (Y/N)")
        if (reply == 'N'):
            break
        else:
            run_genome(winner, config, "human")

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config')
    run(config_path)
