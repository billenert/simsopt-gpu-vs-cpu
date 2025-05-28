import os
import sys
import numpy as np
import argparse
from simsopt.field.boozermagneticfield import (
        BoozerRadialInterpolant,
        InterpolatedBoozerField,
        ShearAlfvenHarmonic,
        ShearAlfvenWavesSuperposition
        )
from simsopt.field.tracing import (
        trace_particles_boozer_perturbed,
        MaxToroidalFluxStoppingCriterion
        )
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        ALPHA_PARTICLE_CHARGE as CHARGE,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
        )
from booz_xform import Booz_xform
from stellgap import AE3DEigenvector
import time

start_time = time.time()

parser = argparse.ArgumentParser(description='Trace lost trajectories with optional parameters')
parser.add_argument('--only_first_10_IC', action='store_true', help='Use only the first 10 initial conditions')
parser.add_argument('-m', '--modemulti', type=float,default=0.0)
args = parser.parse_args()
only_first_10_IC = args.only_first_10_IC
bump_multi = args.modemulti
print("mode multiplier: ", bump_multi)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    single_thread = False
except:
    comm = None
    single_thread = True
first_thread = (comm.rank == 0) or single_thread

max_t_seconds = 1e-3
boozmn_filename = 'boozmn_qhb_100.nc'

saw_file = 'mode/scaled_mode_32.935kHz.npy'
ic_folder = 'initial_conditions/first10' if only_first_10_IC else 'initial_conditions'
s_init = np.loadtxt(f'{ic_folder}/s0.txt', ndmin=1)
equil = Booz_xform()
equil.verbose = False
equil.read_boozmn(boozmn_filename)
nfp = equil.nfp

if first_thread:
    print("Interpolating fields...")

bri = BoozerRadialInterpolant(
        equil=equil,
        order=3,
        no_K=False
)

equil_field = InterpolatedBoozerField(
        field=bri,
        degree=3,
        srange=(0, 1, 15),
        thetarange=(0, np.pi, 15),
        zetarange=(0, 2 * np.pi / nfp, 15),
        extrapolate=True,
        nfp=nfp,
        stellsym=True,
        initialize=['modB','modB_derivs']
)

eigenvector = AE3DEigenvector.load_from_numpy(filename=saw_file)
omega = np.sqrt(eigenvector.eigenvalue)*1000
harmonic_list = []
for harmonic in eigenvector.harmonics:
    sbump = eigenvector.s_coords
    bump = harmonic.amplitudes
    sah = ShearAlfvenHarmonic(
        Phihat_value_or_tuple=(sbump, bump_multi*bump),
        Phim=harmonic.m,
        Phin=harmonic.n,
        omega=omega,
        phase=0.0,
        B0=equil_field
    )
    harmonic_list.append(sah)
saw = ShearAlfvenWavesSuperposition(harmonic_list)

VELOCITY = np.sqrt(2 * ENERGY / MASS)
if first_thread:
    print('Prepare initial conditions')
    s_init = np.loadtxt(f'{ic_folder}/s0.txt', ndmin=1)
    theta_init = np.loadtxt(f'{ic_folder}/theta0.txt', ndmin=1)
    zeta_init = np.loadtxt(f'{ic_folder}/zeta0.txt', ndmin=1)
    vpar_init = np.loadtxt(f'{ic_folder}/vpar0.txt', ndmin=1)
    s_init = s_init[:1]
    theta_init = theta_init[:1]
    zeta_init = zeta_init[:1]
    vpar_init = vpar_init[:1]
    points = np.zeros((s_init.size, 3))
    points[:, 0] = s_init
    points[:, 1] = theta_init
    points[:, 2] = zeta_init
    np.savetxt('points.txt', points)
    saw.B0.set_points(points)
    mu_per_mass = (VELOCITY**2 - vpar_init**2) / (2 * saw.B0.modB()[:,0])
else:
    points = None
    vpar_init = None
    mu_per_mass = None

if not single_thread:
    points = comm.bcast(points, root=0)
    vpar_init = comm.bcast(vpar_init, root=0)
    mu_per_mass = comm.bcast(mu_per_mass, root=0)

if first_thread:
    print('Begin particle tracking...')

gc_tys, gc_hits = trace_particles_boozer_perturbed(
        perturbed_field=saw,
        stz_inits=points,
        parallel_speeds=vpar_init,
        mus=mu_per_mass,
        tmax=max_t_seconds,
        mass=MASS,
        charge=CHARGE,
        Ekin=ENERGY,
        abstol=1e-5,
        reltol=1e-9,
        comm=comm,
        zetas=[],
        omegas=[],
        vpars=[],
        stopping_criteria=[
            MaxToroidalFluxStoppingCriterion(0.9)
        ],
        forget_exact_path=False,
        zetas_stop=False,
        vpars_stop=False,
        mode = 'gc_vac',
        axis=2
        )

if first_thread:
    output_suffix = '_first10' if only_first_10_IC else ''
    output_dir = f'output_nompi{output_suffix}' if single_thread else f'output{output_suffix}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print('Save IDs of lost particles')
    print(f"Total particles tracked: {len(gc_tys)}")
    
    if len(gc_tys) > 0:
        print(f"First particle data shape: {gc_tys[0].shape}")
        print(f"First particle initial state: {gc_tys[0][0, :]}")
        print(f"First particle final state: {gc_tys[0][1, :]}")
    
    timelost = []
    s0 = []
    theta0 = []
    zeta0 = []
    vpar0 = []
    slost = []
    thetalost = []
    zetalost = []
    vparlost = []
    
    for i in range(len(gc_tys)):
        s0.append(gc_tys[i][0,1])
        theta0.append(gc_tys[i][0,2])
        zeta0.append(gc_tys[i][0,3])
        vpar0.append(gc_tys[i][0,4])
        if len(gc_hits[i]):
            timelost.append(gc_hits[i][0,0])
            slost.append(gc_hits[i][0,2])
            thetalost.append(gc_hits[i][0,3])
            zetalost.append(gc_hits[i][0,4])
            vparlost.append(gc_hits[i][0,5])
        else:
            timelost.append(gc_tys[i][1,0])
            slost.append(gc_tys[i][1,1])
            thetalost.append(gc_tys[i][1,2])
            zetalost.append(gc_tys[i][1,3])
            vparlost.append(gc_tys[i][1,4])
    
    np.savetxt(f'{output_dir}/timelost.txt', timelost)
    np.savetxt(f'{output_dir}/s0.txt', s0)
    np.savetxt(f'{output_dir}/theta0.txt', theta0)
    np.savetxt(f'{output_dir}/zeta0.txt', zeta0)
    np.savetxt(f'{output_dir}/vpar0.txt', vpar0)
    np.savetxt(f'{output_dir}/slost.txt', slost)
    np.savetxt(f'{output_dir}/thetalost.txt', thetalost)
    np.savetxt(f'{output_dir}/zetalost.txt', zetalost)
    np.savetxt(f'{output_dir}/vparlost.txt', vparlost)
if first_thread:
    print("All done.")
    print(f"Processed {len(gc_tys)} particles")
    print(f"Results saved to {output_dir}")
    np.save('gc_tys.npy', np.array(gc_tys, dtype=object), allow_pickle=True)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime {runtime} seconds")
