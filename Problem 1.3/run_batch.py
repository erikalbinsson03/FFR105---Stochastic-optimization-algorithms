import math
from genetic_algorithm import run_function_optimization
import matplotlib.pyplot as plt

number_of_runs = 100                # Do NOT change
population_size = 100               # Do NOT change
maximum_variable_value = 5          # Do NOT change: (x_i in [-a,a], where a = maximumVariableValue)
number_of_genes = 50                # Do NOT change
number_of_variables = 2  	         # Do NOT change
tournament_size = 2                 # Do NOT change
tournament_probability = 0.75       # Do NOT change
crossover_probability = 0.8         # Do NOT change
number_of_generations = 2000        # Do NOT change


mutation_probability_list = [0, 0.005, 0.01, 0.02, 0.05, 0.1] # Add more values in this list; see the problem sheet

# Below you should add the required code for the statistical analysis 
# (computing median fitness values, and so on), as described in the problem sheet
median_fitness_values = []
for mutation_probability in mutation_probability_list:
   print("=====================================")
   print(f"mutation probability: {mutation_probability:.3f}")
   print("=====================================")

   fitness_values = []

   for run_index in range(number_of_runs):
      [maximum_fitness, x_best] = run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, tournament_size, \
                                       tournament_probability, crossover_probability, mutation_probability, number_of_generations)
      fitness_values.append(maximum_fitness)

      output = f"Run index: {run_index}, fitness: {maximum_fitness:.4f}, x = ({x_best[0]:.8f},{x_best[1]:.8f})"
      print(output)

   sorted_fitness_values = sorted(fitness_values)
   median_fitness = (sorted_fitness_values[49] + sorted_fitness_values[50]) / 2
   median_fitness_values.append(median_fitness)

   print(f"Median fitness for mutation probability {mutation_probability:.3f}: {median_fitness:.4f}")

print("=====================================")
print(median_fitness_values)

plt.plot(mutation_probability_list, median_fitness_values, marker='o')
plt.xlabel("Mutation Probability")
plt.ylabel("Median Fitness")
plt.show()