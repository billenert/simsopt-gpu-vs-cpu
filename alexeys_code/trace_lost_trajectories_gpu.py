import os
import logging
import numpy as np
import time
from booz_xform import Booz_xform

from simsopt.field import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    trace_particles_boozer,
    MinToroidalFluxStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    compute_resonances
)

from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )

import simsoptpp as sopp

filename = 'boozmn_qhb_100.nc'
ic_folder = 'initial_conditions'

saw_filename = 'mode/scaled_mode_32.935kHz.npy'
saw_data = np.load(saw_filename, allow_pickle=True)
saw_data = saw_data[()]
saw_omega = 1000*np.sqrt(saw_data['eigenvalue'])
print("omega=", saw_omega)
s = saw_data['s_coords']
saw_srange = (s[0], s[-1], len(s))
saw_m = np.ascontiguousarray([x[0] for x in saw_data['harmonics']])
saw_n = np.ascontiguousarray([x[1] for x in saw_data['harmonics']])
saw_phihats = np.ascontiguousarray(np.column_stack([x[2].T for x in saw_data['harmonics']]))
saw_nharmonics = len(saw_m)

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')

s_init = np.loadtxt(f'{ic_folder}/s0.txt', ndmin=1)
theta_init = np.loadtxt(f'{ic_folder}/theta0.txt', ndmin=1)
zeta_init = np.loadtxt(f'{ic_folder}/zeta0.txt', ndmin=1)
vpar_init = np.loadtxt(f'{ic_folder}/vpar0.txt', ndmin=1)
points = np.zeros((s_init.size, 3))
points[:, 0] = s_init
points[:, 1] = theta_init
points[:, 2] = zeta_init
points = np.ascontiguousarray(points)
vpar_init = np.ascontiguousarray(vpar_init)

t1 = time.time()
equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(filename)
nfp = equil.nfp

bri = BoozerRadialInterpolant(
    equil=equil,
    order=3,
    no_K=False
)

degree = 3
srange = (0, 1, 15)
thetarange = (0, np.pi, 15)
zetarange = (0, 2*np.pi/nfp, 15)

field = InterpolatedBoozerField(
    bri,
    degree=3,
    srange=(0, 1, 15),
    thetarange=(0, np.pi, 15),
    zetarange=(0, 2*np.pi/nfp, 15),
    extrapolate=True,
    nfp=nfp,
    stellsym=True,
    initialize=['modB','modB_derivs']
)

# Evaluate error in interpolation
print('Error in |B| interpolation', 
    field.estimate_error_modB(1000),
    flush=True)

VELOCITY = np.sqrt(2*ENERGY/MASS)

# set up GPU interpolation grid
def gen_bfield_info(field, srange, trange, zrange):

	s_grid = np.linspace(srange[0], srange[1], srange[2])
	theta_grid = np.linspace(trange[0], trange[1], trange[2])
	zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

	quad_pts = np.empty((srange[2]*trange[2]*zrange[2], 3))
	for i in range(srange[2]):
		for j in range(trange[2]):
			for k in range(zrange[2]):
				quad_pts[trange[2]*zrange[2]*i + zrange[2]*j + k, :] = [s_grid[i], theta_grid[j], zeta_grid[k]]


	field.set_points(quad_pts)
	G = field.G()
	iota = field.iota()
	diotads = field.diotads()
	I = field.I()
	modB = field.modB()
	J = (G + iota*I)/(modB**2)
	maxJ = np.max(J) # for rejection sampling

	psi0 = field.psi0

	# Build interpolation data
	modB_derivs = field.modB_derivs()

	dGds = field.dGds()
	dIds = field.dIds()

	quad_info = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads))
	quad_info = np.ascontiguousarray(quad_info)

	return quad_info, maxJ, psi0

# generate grid with 15 simsopt grid pts
n_grid_pts = 15
srange = (0, 1, 3*n_grid_pts+1)
trange = (0, np.pi, 3*n_grid_pts+1)
zrange = (0, 2*np.pi/nfp, 3*n_grid_pts+1)
quad_info, maxJ, psi0 = gen_bfield_info(field, srange, trange, zrange)

nparticles = len(points)
print("tracing particles")

last_time = sopp.gpu_tracing_saw(
	quad_pts=quad_info, 
	srange=srange,
	trange=trange,
	zrange=zrange, 
	stz_init=points,
	m=MASS, 
	q=CHARGE, 
	vtotal=VELOCITY,  
	vtang=vpar_init, 
	tmax=1e-3, 
	tol=1e-9, 
	psi0=psi0, 
	nparticles=nparticles,
	saw_srange=saw_srange,
	saw_m=saw_m,
	saw_n=saw_n,
	saw_phihats=saw_phihats,
	saw_omega=saw_omega,
	saw_nharmonics=saw_nharmonics)

last_time = np.reshape(last_time, (nparticles, 7))

results = {
    'timelost' : [],
    'slost' : [],
    'thetalost' : [],
    'zetalost' : [],
    'vparlost' : []
}

results['timelost'] = last_time[:,4]
results['slost'] = last_time[:,0]
results['thetalost'] = last_time[:,1]
results['zetalost'] = last_time[:,2]
results['vparlost'] = last_time[:,3]

if not os.path.exists('output_gpu'):
    os.makedirs('output_gpu')
for name, array in results.items():
    np.savetxt(f'output_gpu/{name}.txt', array)
    
did_leave = [t < 1e-3 for t in results['timelost']]

loss_frac = sum(did_leave) / len(did_leave)
print(f"Number of particles= {nparticles}")
print(f"Loss fraction: {loss_frac:.3f}")
