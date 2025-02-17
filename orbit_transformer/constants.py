import numpy as np

SEED = 1234
RNG = np.random.default_rng(SEED)

KILOMETER = 1_000.0
MINUTE = 60.0
HOUR = 60 * MINUTE
DAY = 24 * HOUR

R_EARTH = 6_371 * KILOMETER
MIN_R_MAG = R_EARTH
MAX_R_MAG = R_EARTH + 69_000 * KILOMETER
MIN_V_MAG = 0.0
MAX_V_MAG = 12 * KILOMETER

MIN_THRUST = 0.0
MAX_THRUST = KILOMETER

MIN_TIME_STEP_SEC = MINUTE
MAX_TIME_STEP_SEC = HOUR

# RANDOM_EARTH_X = list()
# RANDOM_EARTH_Y = list()
# RANDOM_EARTH_Z = list()
# for _ in range(1_000):
#     azimuthal_angle = np.random.uniform(0, 2 * np.pi)
#     polar_angle = np.random.uniform(0, np.pi)
#     x = R_EARTH * np.sin(polar_angle) * np.cos(azimuthal_angle)
#     y = R_EARTH * np.sin(polar_angle) * np.sin(azimuthal_angle)
#     z = R_EARTH * np.cos(polar_angle)
#     RANDOM_EARTH_X.append(x)
#     RANDOM_EARTH_Y.append(y)
#     RANDOM_EARTH_Z.append(z)
