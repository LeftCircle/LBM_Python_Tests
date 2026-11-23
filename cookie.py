import numpy as np

N_LATTICE_POINTS = 9

default_weights = np.array([
	4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 
	1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 
])

dx = 1.0
dt = 1.0

c = [
	np.array([0, 0]),
	np.array([1, 0]),
	np.array([0, 1]),
	np.array([-1, 0]),
	np.array([0, -1]),
	np.array([1, 1]),
	np.array([-1, 1]),
	np.array([-1, -1]),
	np.array([1, -1])
]

cookie_weights = np.array([
	13.0 / 18, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
	1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0 
])

def get_speed_of_sound_squared(dt: float, tau: float) -> float:
	return 1.0 / 3.0 * dx**2 / dt**2

def first_moment(weights: np.ndarray) -> float:
	return sum(weights)

def second_moment(weights: np.ndarray, c: np.ndarray) -> float:
	result = 0
	for i in range(N_LATTICE_POINTS):
		result += weights[i] * c[i][0]
		result += weights[i] * c[i][1]
	return result


def third_moment(weights: np.ndarray, c: np.ndarray, alpha: int, beta: int) -> float:
	result = 0
	for i in range(N_LATTICE_POINTS):
		result += weights[i] * c[i][alpha] * c[i][beta]
	return result 	

cs_sq = get_speed_of_sound_squared(dx, dt)
print("First moment = ", first_moment(cookie_weights))
print("Second moment = ", second_moment(cookie_weights, c))
combinations = [[0, 0], [0, 1], [1, 0], [1, 1]]
print("cs_sq = ", cs_sq)
for comb in combinations:
	print("Third moment alpha {} beta {} = {}".format(comb[0], comb[1], third_moment(default_weights, c, comb[0], comb[1])))




