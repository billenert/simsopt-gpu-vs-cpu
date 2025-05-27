from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer_perturbed, MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion
from simsopt.util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from simsopt.mhd import Vmec
import time
import os
import sys
import numpy as np
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from booz_xform import Booz_xform
from simsopt.util.mpi import MpiPartition
from simsopt._core.util import parallel_loop_bounds

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None
import builtins
sys.stdout = open("stdout.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)
import time
time1 = time.time()

boozmn_filename = 'boozmn_QA_bootstrap.nc'
nParticles = 5000 # Number of particles
ns_interp = 30
ntheta_interp = 45
nzeta_interp = 45
ntheta_min = 100
nzeta_min = 100
ns_min = 100

## Setup Booz_xform object
equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(boozmn_filename)
nfp = equil.nfp

## Call boozxform and setup radial interpolation
order = 3
mpi = MpiPartition(comm_world=comm)
bri = BoozerRadialInterpolant(equil,order,no_K=True,mpi=mpi)

## Setup 3d interpolation
degree = 3
srange = (0, 1, ns_interp)
thetarange = (0, np.pi, ntheta_interp)
zetarange = (0, 2*np.pi/nfp, nzeta_interp)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

Ekin=FUSION_ALPHA_PARTICLE_ENERGY
mass=ALPHA_PARTICLE_MASS
charge=ALPHA_PARTICLE_CHARGE # Alpha particle charge
vpar0=np.sqrt(2*Ekin/mass)

time2 = time.time()
# Compute min/max values of Jacobian
s_grid = np.linspace(0,1,ns_min)
theta_grid = np.linspace(0,2*np.pi,ntheta_min,endpoint=False)
zeta_grid = np.linspace(0,2*np.pi,nzeta_min,endpoint=False)
[zeta_grid,theta_grid,s_grid] = np.meshgrid(zeta_grid,theta_grid,s_grid)
points = np.zeros((len(theta_grid.flatten()),3))
points[:,0] = s_grid.flatten()
points[:,1] = theta_grid.flatten()
points[:,2] = zeta_grid.flatten()
field.set_points(points)
G = field.G()
iota = field.iota()
I = field.I()
modB = field.modB()
J = (G + iota*I)/(modB**2)
# minJ = np.min(J)
maxJ = np.max(J)

theta_init = []
zeta_init = []
s_init = []
points = np.zeros((1,3))
# Initialize particles uniformaly wrt volume element

first, last = parallel_loop_bounds(comm,nParticles)
for i in range(first,last):
        while True:
            rand1 = np.random.uniform(0,1,None)
            s = np.random.uniform(0,1.0,None)
            theta = np.random.uniform(0,2*np.pi,None)
            zeta  = np.random.uniform(0,2*np.pi/nfp,None)
            points[:,0] = s
            points[:,1] = theta
            points[:,2] = zeta
            field.set_points(points)
            J = (field.G()[0,0] + field.iota()[0,0]*field.I()[0,0])/(field.modB()[0,0]**2)
            J = J/maxJ # Normalize

            if (rand1 <= J):
                s_init.append(s)
                theta_init.append(theta)
                zeta_init.append(zeta)
                break

s_init = [i for o in comm.allgather(s_init) for i in o]
theta_init = [i for o in comm.allgather(theta_init) for i in o]
zeta_init = [i for o in comm.allgather(zeta_init) for i in o]

if comm.rank == 0:
    vpar_init = np.random.uniform(-vpar0,vpar0,(nParticles,))

    print("Elapsed time for computing points = "+str(time2-time1))
    time1 = time.time()
    points = np.zeros((nParticles,3))
    points[:,0] = s_init
    points[:,1] = np.asarray(theta_init)
    points[:,2] = np.asarray(zeta_init)
    field.set_points(points)
    mu_init = (vpar0**2 - vpar_init**2)/(2*field.modB()[0,0])

    np.savetxt('s0.txt',s_init)
    np.savetxt('theta0.txt',theta_init)
    np.savetxt('zeta0.txt',zeta_init)
    np.savetxt('mu0.txt',mu_init)
    np.savetxt('vpar0.txt',vpar_init)
