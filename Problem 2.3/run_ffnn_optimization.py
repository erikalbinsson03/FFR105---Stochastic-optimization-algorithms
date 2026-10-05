import math
import random
import matplotlib.pyplot as plt

from run_encoding_decoding_test import decode_chromosome
from slopes import get_slope_angle


N_INPUTS = 3
N_HIDDEN = 5
N_OUTPUTS = 2

W_MAX = 1.0
C = 2.0

g = 9.81

M = 20000
tau = 30
C_h = 40
T_amb = 283
C_b = 3000

T_max = 750
v_max = 25
v_min = 1
alpha_max = 10

dt = 0.1

POPULATION_SIZE = 50
MAX_GENERATIONS = 200

CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
MUTATION_SIZE = 0.1

TOURNAMENT_SIZE = 3

TRAINING_SLOPES = 10
VALIDATION_SLOPES = 5


def sigmoid(x, c):

    return 1.0 / (1.0 + math.exp(-c * x))


def run_ffnn(chromosome, v, alpha_deg, T_b):

    input_values = [v / v_max, alpha_deg / alpha_max, T_b / T_max]

    w_ih, w_ho = decode_chromosome(chromosome, N_INPUTS, N_HIDDEN, N_OUTPUTS, W_MAX)

    inputs = input_values + [1.0]

    hidden = []

    for h in range(N_HIDDEN):

        total = 0.0

        for i in range(N_INPUTS + 1):

            total += w_ih[h][i] * inputs[i]

        hidden.append(sigmoid(total, C))

    hidden_with_bias = hidden + [1.0]

    raw_outputs = []

    for o in range(N_OUTPUTS):

        total = 0.0

        for h in range(N_HIDDEN + 1):

            total += w_ho[o][h] * hidden_with_bias[h]

        raw_outputs.append(total)

    P_p = sigmoid(raw_outputs[0], C)

    gear_output = sigmoid(raw_outputs[1], C)

    if gear_output < 1.0 / 3.0:

        delta_gear = -1

    elif gear_output > 2.0 / 3.0:

        delta_gear = 1

    else:

        delta_gear = 0

    return P_p, delta_gear


def simulate_truck(chromosome, slope_index, data_set_index):

    x = 0.0
    v = 20.0
    T_b = 500.0
    gear = 7

    time_since_gear_change = 2.0

    positions = []
    speeds = []

    engine_brake_factors = [7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.4, 1.2, 1.0]

    while x < 1000:

        alpha_deg = get_slope_angle(x, slope_index, data_set_index)
        alpha = math.radians(alpha_deg)

        P_p, delta_gear = run_ffnn(chromosome, v, alpha_deg, T_b)

        if time_since_gear_change >= 2.0:

            if delta_gear != 0:

                gear += delta_gear
                gear = max(1, min(10, gear))

                time_since_gear_change = 0.0

        F_g = M * g * math.sin(alpha)

        if T_b < T_max - 100:

            F_b = (M * g / 20) * P_p

        else:

            F_b = ((M * g / 20) * P_p * math.exp(-(T_b - (T_max - 100)) / 100))

        F_eb = engine_brake_factors[gear - 1] * C_b

        acceleration = (F_g - F_b - F_eb) / M

        v_old = v
        v += acceleration * dt

        x += v_old * dt

        delta_T_b = T_b - T_amb

        if P_p < 0.01:

            d_delta_T_b = -delta_T_b / tau

        else:

            d_delta_T_b = C_h * P_p

        delta_T_b += d_delta_T_b * dt
        T_b = T_amb + delta_T_b

        positions.append(x)
        speeds.append(v)

        time_since_gear_change += dt

        if v > v_max:
            break

        if v < v_min:
            break

        if T_b > T_max:
            break

    return positions, speeds


def calculate_fitness(chromosome, slope_index, data_set_index):

    positions, speeds = simulate_truck(chromosome, slope_index, data_set_index)

    if len(positions) == 0:
        return 0.0

    distance = positions[-1]
    average_speed = sum(speeds) / len(speeds)

    return average_speed * distance


def evaluate_chromosome(chromosome, data_set_index, number_of_slopes):

    fitness_values = []

    for slope_index in range(1, number_of_slopes + 1):

        fitness = calculate_fitness(chromosome, slope_index, data_set_index)
        fitness_values.append(fitness)

    return sum(fitness_values) / len(fitness_values)


def create_random_chromosome():

    chromosome_length = (N_HIDDEN * (N_INPUTS + 1) + N_OUTPUTS * (N_HIDDEN + 1))
    chromosome = []

    for i in range(chromosome_length):

        chromosome.append(random.random())

    return chromosome


def create_initial_population():

    population = []

    for i in range(POPULATION_SIZE):

        population.append(create_random_chromosome())

    return population


def tournament_selection(population, fitnesses):

    selected = []

    for i in range(TOURNAMENT_SIZE):

        index = random.randrange(len(population))
        selected.append(index)

    best_index = selected[0]

    for index in selected[1:]:

        if fitnesses[index] > fitnesses[best_index]:

            best_index = index

    return population[best_index]


def crossover(parent_1, parent_2):

    if random.random() > CROSSOVER_RATE:

        return parent_1[:], parent_2[:]

    point = random.randint(1, len(parent_1) - 1)

    child_1 = (parent_1[:point] + parent_2[point:])
    child_2 = (parent_2[:point] + parent_1[point:])

    return child_1, child_2


def mutate(chromosome):

    for i in range(len(chromosome)):

        if random.random() < MUTATION_RATE:

            chromosome[i] += random.gauss(0, MUTATION_SIZE)
            chromosome[i] = max(0.0, min(1.0, chromosome[i]))

    return chromosome


def create_new_population(population, fitnesses):

    new_population = []
    best_index = 0

    for i in range(1, len(population)):

        if fitnesses[i] > fitnesses[best_index]:

            best_index = i

    new_population.append(population[best_index][:])

    while len(new_population) < POPULATION_SIZE:

        parent_1 = tournament_selection(population, fitnesses)
        parent_2 = tournament_selection(population, fitnesses)

        child_1, child_2 = crossover(parent_1, parent_2)
        child_1 = mutate(child_1)
        child_2 = mutate(child_2)

        new_population.append(child_1)

        if len(new_population) < POPULATION_SIZE:

            new_population.append(child_2)

    return new_population


def save_best_chromosome(chromosome):

    with open("best_chromosome.py", "w") as file:

        file.write("best_chromosome = " + repr(chromosome) + "\n")


def run_ga():

    population = create_initial_population()

    training_history = []
    validation_history = []

    best_chromosome = None
    best_validation_fitness = float("-inf")
    generations_without_improvement = 0

    for generation in range(MAX_GENERATIONS):

        training_fitnesses = []

        for chromosome in population:
            fitness = evaluate_chromosome(chromosome, 1, TRAINING_SLOPES)
            training_fitnesses.append(fitness)

        validation_fitnesses = []

        for chromosome in population:
            fitness = evaluate_chromosome(chromosome, 2, VALIDATION_SLOPES)
            validation_fitnesses.append(fitness)

        generation_best_index = 0

        for i in range(1, len(population)):

            if training_fitnesses[i] > training_fitnesses[generation_best_index]:

                generation_best_index = i

        generation_training_fitness = training_fitnesses[generation_best_index]
        generation_validation_fitness = validation_fitnesses[generation_best_index]

        training_history.append(generation_training_fitness)
        validation_history.append(generation_validation_fitness)

        if generation_validation_fitness > best_validation_fitness:

            best_validation_fitness = generation_validation_fitness
            best_chromosome = population[generation_best_index][:]
            generations_without_improvement = 0

        else:

            generations_without_improvement += 1

        print(
            "Generation:", generation + 1,
            "Training:", generation_training_fitness,
            "Validation:", generation_validation_fitness
        )

        population = create_new_population(
            population,
            training_fitnesses
        )

        if generations_without_improvement >= 30:
            break

    save_best_chromosome(best_chromosome)

    plt.figure()

    plt.plot(
        range(1, len(training_history) + 1),
        training_history,
        label="Training"
    )

    plt.plot(
        range(1, len(validation_history) + 1),
        validation_history,
        label="Validation"
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    return best_chromosome


if __name__ == "__main__":

    best_chromosome = run_ga()

    print("Best chromosome:")
    print(best_chromosome)