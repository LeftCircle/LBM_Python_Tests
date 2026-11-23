import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



c_s = 1.0 / np.sqrt(3.0)
print(c_s)
one_over_cs_squared = 1.0 / (c_s**2)
one_over_cs_fourth = one_over_cs_squared**2
dt = 1.0
tau = 0.9

weights = np.array([
	4.0 / 9.0,
	1.0 / 9.0,
	1.0 / 9.0,
	1.0 / 9.0,
	1.0 / 9.0,
	1.0 / 36.0,
	1.0 / 36.0,
	1.0 / 36.0,
	1.0 / 36.0
])

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

nx = 30
ny = 30

densities = np.ones((nx, ny))
densities *= 1
densities[int(nx / 2)][int(ny / 2)] = 10
ux = np.zeros((nx, ny))
uy = np.zeros((nx, ny))

#fis = np.tile(weights, (nx, ny, 1))
fis = np.zeros((nx, ny, 9))
feq = np.zeros((nx, ny, 9))
fstar = np.zeros((nx, ny, 9))


def init():
	for j in range(ny):
		for i in range(nx):
			for k in range(9):
				fis[i, j, k] = fi_eq(densities[i, j], np.array([0.0, 0.0]), k)

def get_dens(f_i : np.array) -> float:
	# lbm points should be a 3x3 matrix
	return np.sum(f_i)

def get_u(f_i, dens) -> np.array:
	u = np.array([0.0, 0.0])
	u[0] = (f_i[1] + f_i[5] + f_i[8] - (f_i[3] + f_i[6] + f_i[7])) / dens
	u[1] = (f_i[2] + f_i[5] + f_i[6] - (f_i[4] + f_i[7] + f_i[8])) / dens
	return u

def fi_eq(dens, u, i) -> float:
	cu = np.dot(c[i], u)
	uu = np.dot(u, u)
	return weights[i] * dens * (
		1.0 + cu * one_over_cs_squared +
		cu**2 * 0.5 * one_over_cs_fourth -
		uu * 0.5 * one_over_cs_squared
	)

def get_fstar(f, feq):
	return f * ( 1.0 - dt / tau) + feq * dt / tau


def lbm_step():
	for j in range(ny):
		for i in range(nx):
			# moment update
			densities[i][j] = get_dens(fis[i][j])
			vel = get_u(fis[i, j], densities[i][j])
			ux[i][j] = vel[0]
			uy[i][j] = vel[1]
			# equilibrium
			for k in range(9):
				feq[i][j][k] = fi_eq(densities[i][j], vel, k)
	# could output here?

	# collision
	for j in range(ny):
			for i in range(nx):
				for k in range(9):
					fstar[i][j][k] = get_fstar(fis[i][j][k], feq[i][j][k])

	# propagation. Transfer f_is to neighbor cells based on c_i
	for j in range(ny):
		for i in range(nx):
			for k in range(9):
				ni = (i + c[k][0]) % nx
				nj = (j + c[k][1]) % ny
				fis[ni][nj][k] = fstar[i][j][k]


# --------------------------------------------------------------------------
# NOTE -> Visualization stff from claude

def run_and_animate(n_steps=100, vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(densities.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    title = ax.set_title('Density (step 0)')

    def update(frame):
        lbm_step()
        im.set_data(densities.T)      # same vmin/vmax every frame, no re-scaling
        title.set_text(f'Density (step {frame})')
        return im, title

    anim = FuncAnimation(fig, update, frames=n_steps, interval=100)
    anim.save('diffusion.gif', writer='pillow', fps=10)
    print("Saved to diffusion.gif")
# --------------------------------------------------------------------------


def main():
	init()
	#print(densities)
	run_and_animate(50)



if __name__ == "__main__":
	main()