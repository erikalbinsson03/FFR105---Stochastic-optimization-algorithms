import numpy as np
import matplotlib.pyplot as plt

NUMBER_ITERATIONS = 200

def objective_function(x):

    x1 = x[0]
    x2 = x[1]

    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2

def run_PSO_algorithm(objective, number_particles, number_dimensions, x_min, x_max, alpha, delta_t, c_1, c_2, w_initial, beta, w_min, v_max):

    positions = np.random.uniform(x_min, x_max, size=(number_particles, number_dimensions))
    velocities = alpha / delta_t * (-(x_max - x_min) / 2 + np.random.uniform(0, 1, size=(number_particles, number_dimensions)) * (x_max - x_min))

    values = np.array([objective(position) for position in positions])
    particle_best_positions = positions.copy()
    particle_best_values = values.copy()

    best_index = np.argmin(particle_best_values)
    swarm_best_position = particle_best_positions[best_index].copy()
    swarm_best_value = particle_best_values[best_index].copy()

    w = w_initial

    for iteration in range(NUMBER_ITERATIONS):

        q = np.random.uniform(0, 1, size=(number_particles, 1))
        r = np.random.uniform(0, 1, size=(number_particles, 1))

        velocities = (w * velocities + c_1 * q * (particle_best_positions - positions) / delta_t + c_2 * r * (swarm_best_position - positions) / delta_t)
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities * delta_t
        values = np.array([objective(position) for position in positions])

        improved = values < particle_best_values
        particle_best_positions[improved] = positions[improved]
        particle_best_values[improved] = values[improved]

        best_index = np.argmin(particle_best_values)

        if particle_best_values[best_index] < swarm_best_value:

            swarm_best_position = particle_best_positions[best_index].copy()
            swarm_best_value = particle_best_values[best_index]

        w = max(w * beta, w_min)

    return swarm_best_position, swarm_best_value


number_particles = 30
number_dimensions = 2

x_min = -5
x_max = 5

alpha = 1.0
delta_t = 1.0
c_1 = 2.0
c_2 = 2.0

w_initial = 1.4
beta = 0.99
w_min = 0.4

v_max = 1.0

number_runs = 50

for run in range(number_runs):

    best_position, best_value = run_PSO_algorithm(
        objective_function,
        number_particles,
        number_dimensions,
        x_min,
        x_max,
        alpha,
        delta_t,
        c_1,
        c_2,
        w_initial,
        beta,
        w_min,
        v_max
    )

    print(f"Run {run + 1}: x1 = {best_position[0]:.6f}, x2 = {best_position[1]:.6f}, f = {best_value:.6e}")

x1 = np.linspace(-5, 5, 500)
x2 = np.linspace(-5, 5, 500)

X1, X2 = np.meshgrid(x1, x2)

F = np.zeros(X1.shape)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        F[i, j] = objective_function(x)

Z = np.log(0.01 + F)

plt.figure()

contours = plt.contour(X1, X2, Z, levels=30)

plt.clabel(contours)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("$\\log(0.01 + f(x_1,x_2))$")

plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.show()
