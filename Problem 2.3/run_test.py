import math
import matplotlib.pyplot as plt

from best_chromosome import best_chromosome
from run_ffnn_optimization import run_ffnn
from slopes import get_slope_angle


M = 20000
g = 9.81

tau = 30
C_h = 40
T_amb = 283
C_b = 3000

T_max = 750
v_max = 25
v_min = 1

dt = 0.1

slope_index = 1
data_set_index = 3

x = 0.0
v = 20.0
T_b = 500.0
gear = 7

time_since_gear_change = 2.0

engine_brake_factors = [7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.4, 1.2, 1.0]

positions = []
slope_angles = []
brake_pressures = []
gears = []
speeds = []
brake_temperatures = []

while x < 1000:

    alpha_deg = get_slope_angle(x, slope_index, data_set_index)

    alpha = math.radians(alpha_deg)

    P_p, delta_gear = run_ffnn(best_chromosome, v, alpha_deg, T_b)

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
    slope_angles.append(alpha_deg)
    brake_pressures.append(P_p)
    gears.append(gear)
    speeds.append(v)
    brake_temperatures.append(T_b)

    time_since_gear_change += dt

    if v > v_max:
        break

    if v < v_min:
        break

    if T_b > T_max:
        break


fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

axes[0].plot(positions, slope_angles)
axes[0].set_ylabel("Slope angle α [deg]")
axes[0].grid(True)

axes[1].plot(positions, brake_pressures)
axes[1].set_ylabel("Brake pressure")
axes[1].grid(True)

axes[2].step(positions, gears, where="post")
axes[2].set_ylabel("Gear")
axes[2].grid(True)

axes[3].plot(positions, speeds)
axes[3].set_ylabel("Speed [m/s]")
axes[3].grid(True)

axes[4].plot(positions, brake_temperatures)
axes[4].set_ylabel("Brake temperature [K]")
axes[4].set_xlabel("Horizontal distance x [m]")
axes[4].grid(True)

plt.tight_layout()
plt.show()
