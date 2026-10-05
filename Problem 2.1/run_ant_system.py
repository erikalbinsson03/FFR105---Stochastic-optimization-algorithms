###########################
#
# Ant System (AS) for TSP
#
###########################

import math
import numpy as np
import matplotlib.pyplot as plt
import random

def initialize_pheromone_levels(number_of_cities, tau_0):

  pheromone_levels = np.full((number_of_cities, number_of_cities), tau_0)

  return pheromone_levels

def get_visibility(city_locations):

  eta = []
  for city in city_locations:
    eta_ij = []

    for other_city in city_locations:
      if city != other_city:
        distance = np.sqrt((city[0] - other_city[0]) ** 2 + (city[1] - other_city[1]) ** 2)
        eta_ij.append(1 / distance)
      else:
        eta_ij.append(0)
    eta.append(eta_ij)
  return np.array(eta)


def get_node(taboo, pheromone_levels, visibility, alpha, beta):

  probabilities = []
  current_city = taboo[-1]

  denominator = 0

  for j in range(number_of_cities):
    if j not in taboo:
      denominator += (pheromone_levels[current_city][j] ** alpha * visibility[current_city][j] ** beta)

  for j in range(number_of_cities):
    if j in taboo:
      probabilities.append(0)
    else:
      probabilities.append((pheromone_levels[current_city][j] ** alpha * visibility[current_city][j] ** beta) / (denominator))

  next_city = np.random.choice(number_of_cities, p = probabilities)
  
  return next_city

def generate_path(pheromone_levels, visibility, alpha, beta):

  taboo = []
  starting_node = random.randrange(0, number_of_cities)
  taboo.append(starting_node)

  while len(taboo) != number_of_cities:
    next_node = get_node(taboo, pheromone_levels, visibility, alpha, beta)
    taboo.append(next_node)
  return taboo

def get_path_length(path, city_locations):

  length = 0
  for i in range(len(path)):
    if i != len(path) - 1:
      length += np.sqrt((city_locations[path[i]][0] - city_locations[path[i + 1]][0]) ** 2 + (city_locations[path[i]][1] - city_locations[path[i + 1]][1]) ** 2)

    else: 
      length += np.sqrt((city_locations[path[i]][0] - city_locations[path[0]][0]) ** 2 + (city_locations[path[i]][1] - city_locations[path[0]][1]) ** 2)

  return length

def compute_delta_pheromone_levels(path_collection, path_length_collection):

  delta_pheromone_levels = np.zeros((number_of_cities, number_of_cities))
  for path, path_length_collection in zip(path_collection, path_length_collection):
    
    for i in range(len(path)):

      if i != len(path) - 1:
        delta_pheromone_levels[path[i]][path[i + 1]] += 1 / path_length

      else:
        delta_pheromone_levels[path[i]][path[0]] += 1 / path_length

  return delta_pheromone_levels

def update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho):

  pheromone_levels = (1 - rho) * pheromone_levels + delta_pheromone_levels

  pheromone_levels = np.maximum(pheromone_levels, 1e-15)

  return pheromone_levels

##################################################
#  Plots the cities (nodes):
##################################################

def plot_cities(plt, city_locations):
    x = []
    y = []

    for city in city_locations:
        x.append(city[0])
        y.append(city[1])

    plt.scatter(x, y, color='yellow')


def plot_path(plt, path, city_locations):
    x = []
    y = []

    for city_index in path:
        x.append(city_locations[city_index][0])
        y.append(city_locations[city_index][1])

    # Add the starting city again to close the tour
    x.append(city_locations[path[0]][0])
    y.append(city_locations[path[0]][1])

    plt.plot(x, y, color='lime')


#####################################
# Main program:
#####################################

###########################
# Data:
###########################
from city_data import city_locations
number_of_cities = len(city_locations)

###########################
# Parameters:
###########################
number_of_ants = 50 ## Changes allowed.
alpha = 1.0         ## Changes allowed.
beta = 5.0          ## Changes allowed.
rho = 0.5           ## Changes allowed.
tau_0 = 0.1         ## Changes allowed.

target_path_length = 99.9999999

#################################
# Initialization:
#################################

plt.xlim(0, 20)
plt.ylim(0, 20)

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_facecolor('black')

plt.ion()

pheromone_levels = initialize_pheromone_levels(number_of_cities, tau_0)
visibility = get_visibility(city_locations)

#################################
# Main loop:
#################################

iteration_index = 0
minimum_path_length = math.inf
path_length = math.inf


while (minimum_path_length > target_path_length):
  iteration_index += 1
  path_collection = []
  path_length_collection = []
  for ant_index in range(number_of_ants):  
    # Generate paths:
    path = generate_path(pheromone_levels, visibility, alpha, beta) # Uncomment after writing the function
    path_length = get_path_length(path, city_locations) # Uncomment after writing the function
    if (path_length < minimum_path_length):
      minimum_path_length = path_length
      best_path = path.copy()
      print(minimum_path_length)
      
      plt.cla()

      plot_cities(plt, city_locations)
      plot_path(plt, best_path, city_locations)

      plt.title(f"Best path length: {minimum_path_length:.2f}")

      plt.pause(0.05)

    path_collection.append(path)
    path_length_collection.append(path_length)
  # Update pheromone levels:
  delta_pheromone_levels = compute_delta_pheromone_levels(path_collection,path_length_collection) # Uncomment after writing the function
  pheromone_levels = update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho) # Uncomment after writing the function

input(f'Press return to exit')

print("Best path:", best_path)
print("Best path length:", minimum_path_length)

with open("best_result_found.py", "w") as file:
    file.write(f"best_path = {best_path}\n")

plt.ioff()
plt.show()