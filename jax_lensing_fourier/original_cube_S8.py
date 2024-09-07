import os,sys
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="true"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Explicitly requested dtype <class 'jax.numpy.int64'>")
warnings.filterwarnings("ignore", category=UserWarning, message="SyntaxWarning: invalid escape sequence")
warnings.filterwarnings("ignore", category=UserWarning, message="The NVIDIA driver's CUDA version is")

import jax
jax.config.update("jax_enable_x64", False)
jax.clear_caches()
from cuboid_remap import remap
from cuboid_remap import remap_Lbox
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random
import jax.numpy.fft as fft
from jax.scipy.ndimage import map_coordinates
import numpyro 
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.handlers import seed, trace, condition
from jaxpm.kernels import fftk
from jaxpm.pm import lpt
from jaxpm.pm import pm_forces, growth_factor, growth_rate
from jaxpm.utils import gaussian_smoothing
from jaxpm.painting import cic_paint, compensate_cic, cic_paint_2d, cic_read
from jaxpm.kernels import gradient_kernel, laplace_kernel, longrange_kernel
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax_cosmo.scipy.integrate import simps
import diffrax
from diffrax import diffeqsolve,Heun, ODETerm, Dopri5, Dopri8,Tsit5, SaveAt, LeapfrogMidpoint, ReversibleHeun, KenCarp5
from functools import partial
import scipy
import pickle
from cuboid_remap.cuboid import Cuboid
from copy import deepcopy

parser  = argparse.ArgumentParser()
parser.add_argument('method'               , default=None, type=str, help='')
parser.add_argument('run'                  , default=None, type=int, help='')
parser.add_argument('resume_state'         , default=None, type=int, help='')
#parser.add_argument('--grid_size'          , nargs=3, type=int, help='Third set of three numbers')
#parser.add_argument('--box_size'           , nargs=3, type=int, help='Third set of three numbers')
parser.add_argument('--cube_size'           , nargs=1, type=int, help='Third set of three numbers')
parser.add_argument('--cubegrid_size'       , nargs=1, type=int, help='Third set of three numbers')
#parser.add_argument('--lensplane_npix'     , nargs=1, type=int, help='Third set of three numbers')
#parser.add_argument('--lensplane_smoothing', nargs=1, type=float, help='Third set of three numbers')
parser.add_argument('--lensplane_width'    , nargs=1, type=int, help='Third set of three numbers')
#parser.add_argument('--output_npix'        , nargs=1, type=int, help='Third set of three numbers')
parser.add_argument('--butterworth'        , nargs=2, type=float, help='Third set of three numbers')
parser.add_argument('--runname'            , nargs=1, type=str, help='Third set of three numbers')
parser.add_argument('--dir_out'            , nargs=1, type=str, help='Third set of three numbers')

args           = parser.parse_args()
method         = args.method
run            = args.run
resume_state   = args.resume_state
#grid_size      = args.grid_size
#box_size       = args.box_size
cube_size       = args.cube_size
cubegrid_size   = args.cubegrid_size
dir_out        = args.dir_out


#output_npix    = args.output_npix
butterworth    = args.butterworth

#density_plane_npix      = args.lensplane_npix
#density_plane_smoothing = args.lensplane_smoothing
density_plane_width     = args.lensplane_width
name                    = args.runname
cube_size               = cube_size[0]
cubegrid_size           = cubegrid_size[0]
dir_out                 = dir_out[0]
#density_plane_npix      = density_plane_npix[0]
#density_plane_smoothing = density_plane_smoothing[0]
density_plane_width     = density_plane_width[0]
#output_npix             = output_npix[0]
name                    = name


# 350/256
u1=(5, 3, 1)
u2=(1, 0, 0)
u3=(0, 1, 0)

# 400/256
#u1=(4, 3, 1)
#u2=(1, 1, 0)
#u3=(0, 1, 0)

# 300/256
#u1=(7, 3, 1)
#u2=(2, 1, 0)
#u3=(0, 0, 1)

L1,L2,L3=remap_Lbox(u1=u1, u2=u2, u3=u3)
box_size=[L1*cube_size,L2*cube_size,L3*cube_size]
print(box_size)
cubegrid_sizeT=[L1*cubegrid_size,L2*cubegrid_size,L3*cubegrid_size]
print("grid_size:",cubegrid_sizeT)

out_dim = [jnp.int32(jnp.ceil(box_size[1])),jnp.int32(jnp.ceil(box_size[2]))]
print(out_dim)
#3 2 2   1 1 1   0 0 1
######## Transformation
C = Cuboid(u1,u2,u3) #4.1231 0.3430 0.7071
transform = jax.vmap(C.Transform, in_axes=0)


#dir='/net/scratch/yomori/sampletest/'

#field_size = jnp.arctan2(box_size[0],box_size[-1])/jnp.pi*180.

field_size = [jnp.arctan2(box_size[1],box_size[0])/jnp.pi*180., jnp.arctan2(box_size[2],box_size[0])/jnp.pi*180.]
print('Field size: %.2fx%.2f=%.2f'%(field_size[0],field_size[1],(field_size[0]*field_size[1])))

zz       = jnp.linspace(0, 1.5, 1000)
mu       = jnp.array([0.85,0.95,1.05]) #0.95
sigma_z  = jnp.array([0.05,0.05,0.05]) #0.025
ngal     = jnp.array([1.00,1.00,1.00])
sigma_e  = jnp.array([0.26,0.26,0.26])
nz       = [(1/(sigma_z[zi]*jnp.sqrt(2*jnp.pi)))*jnp.exp(-0.5*((zz-mu[zi])/sigma_z[zi])**2) for zi in range(3)] 
nz_shear = [jc.redshift.kde_nz(zz,nz[i],bw=zz[2]-zz[1], zmax=3.0, gals_per_arcmin2=ngal[i]) for i in range(3)]


# Generate noise covariance matrix
'''
def filter(pix_size, N, l0, n):
    N     = int(N)
    ones  = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX    = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY    = np.transpose(kX)
    K     = np.sqrt(kX**2. + kY**2.)
    ell2d = K * 2. * np.pi
    l  = np.arange(4001)
    y  = 1/(1+(l/l0)**(2*n) )**0.5 # Butterworth filter
    f  = np.interp(ell2d.flatten(),l,y,right=0).reshape((N,N))
    return f


#filt2d = filter(field_size*60/output_npix,output_npix,butterworth[0],butterworth[1])
#aa     = np.linalg.solve(noisecov, np.eye(noisecov.shape[0]))

print('Computing filter')
filt2d = filter(field_size*60/output_npix,output_npix,butterworth[0],butterworth[1])

print('Computing noisecov')
if ~np.all(sigma_e == sigma_e[0]):
    raise ValueError(f"For now sigma_e has to be same for all bins")
if ~np.all(ngal == ngal[0]):
    raise ValueError(f"For now ngal has to be same for all bins")


file_noisecov='noisecov.npy'

if os.path.exists(file_noisecov):
    noisecov = np.load(file_noisecov)
else:
    ret    = np.zeros((int(output_npix*output_npix),50000))
    for i in range(0,50000):
        tmp      = np.random.normal(jnp.zeros((output_npix ,output_npix )), sigma_e[0]/jnp.sqrt(ngal[0]*(field_size*60/output_npix)**2)) 
        ret[:,i] = fft.ifft2(fft.fft2(tmp)*filt2d).real.flatten()
    noisecov = np.cov(ret)
    np.save('noisecov.npy',noisecov)

#noisecov = np.load(file_noisecov)
print('Done computing noisecov')
'''

'''
ret    = np.zeros((int(output_npix*output_npix),50000))
for i in range(0,50000):
    tmp      = np.random.normal(jnp.zeros((output_npix ,output_npix )), sigma_e[0]/jnp.sqrt(ngal[0]*(field_size*60/output_npix)**2)) 
    ret[:,i] = fft.ifft2(fft.fft2(tmp)*filt2d).real.flatten()
noisecov = np.cov(ret)

np.save('noisecov.py',noisecov)
'''

#scale_tril = scipy.linalg.cholesky(noisecov, lower=True)




def linear_field(mesh_shape, box_size, pk):
    """Generate initial conditions"""
    kvec   = fftk(mesh_shape)
    kmesh  = sum((kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (box_size[0] * box_size[1] * box_size[2])

    field  = numpyro.sample('initial_conditions',dist.Normal(jnp.zeros(mesh_shape), jnp.ones(mesh_shape)))
    field  = jnp.fft.rfftn(field) * pkmesh**0.5
    field  = jnp.fft.irfftn(field)
    return field



def convergence_Born(cosmo,density_planes, r, a, dx, dz, coords, z_source):
    #Compute constant prefactor:
    Clens = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2

    # Compute comoving distance of source galaxies
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    convergence = 0

    n_planes    = len(r)

    def scan_fn(carry, i):
        density_planes, a, r = carry
        p = density_planes[:,:,i]
        p = (p - p.mean()) * Clens * dz * r[i] / a[i] 
        # Interpolate at the density plane coordinates
        im = map_coordinates(p, coords * r[i] / dx - 0.5, order=1, mode="wrap")

        return carry, im * jnp.clip(1. - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

    # Similar to for loops but using a jaxified approach
    _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

    return convergence.sum(axis=0)



def lpt_lightcone(cosmo, initial_conditions, positions, a, mesh_shape):
    """ Computes first order LPT displacement """
    initial_force = pm_forces(positions, delta=initial_conditions).reshape(mesh_shape+[3])
    a  = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a).reshape([1,1,-1,1]) * initial_force
    p  = (a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo, a)) * growth_factor(cosmo, a)).reshape([1,1,-1,1]) * initial_force
    return dx.reshape([-1,3]), p.reshape([-1,3])



def make_full_field_model(field_size, cube_size, cubegrid_size, box_size, out_dim, transform, method='lpt',density_plane_width=None, density_plane_npix=None, gs=None):
  
    def density_plane_fn(t, y, args):

        cosmo, _ , density_plane_width, density_plane_npix,cc  = args

        # Load positions and apply boundary conditions
        positions  = y[0]
        positions = jnp.mod(positions, cubegrid_size) # Apply boundary condition
        
        # Make grid such that 1 grid is 1Mpc/h
        #gx=jnp.ceil(box_size[1])
        #gy=jnp.ceil(box_size[2])
        #gz=jnp.ceil(box_size[0])

        
        #density_plane_npix = [box_size[0],box_size[1]]

        # Width oin units of Mpc/h
        #w = density_plane_width / box_size[0] * gz

        #jax.debug.print("boxsize[0]: {}", box_size[0] )
        #jax.debug.print("w: {}", w)

        # Converts time t to comoving distance in voxel coordinates
        center = jax.numpy.interp(t, jnp.linspace(0.005,1,500), cc) # comoving radial distance in Mpc/h

        #jax.debug.print("cubemax x: {}", jnp.max(positions[:,0]))
        #jax.debug.print("cubemin x: {}", jnp.min(positions[:,0]))
        #jax.debug.print("cubemax y: {}", jnp.max(positions[:,1]))
        #jax.debug.print("cubemin y: {}", jnp.min(positions[:,1]))
        #jax.debug.print("cubemax z: {}", jnp.max(positions[:,2]))
        #jax.debug.print("cubemin z: {}", jnp.min(positions[:,2]))

        # Note -- position here is in terms of grid size not box size 
        #jax.debug.breakpoint()
        positions = transform(positions/cubegrid_size)*cube_size
        
        #jax.debug.print("---------post transform--------")

        #jax.debug.print("max z: {}", jnp.max(positions[:,0]))
        #jax.debug.print("min z: {}", jnp.min(positions[:,0]))
        #jax.debug.print("max x: {}", jnp.max(positions[:,1]))
        #jax.debug.print("min x: {}", jnp.min(positions[:,1]))
        #jax.debug.print("max y: {}", jnp.max(positions[:,2]))
        #jax.debug.print("min y: {}", jnp.min(positions[:,2]))
        

        #positions = positions[:, [2, 1, 0]]

        x      = positions[..., 1]/box_size[1] * density_plane_npix[0]
        y      = positions[..., 2]/box_size[2] * density_plane_npix[1]
        d      = positions[..., 0]
        
        #jax.debug.print("NUMPART : {}", len(x))
        # Selecting only particles that fall inside the volume of interest
        #jax.debug.print("dmin : {}", d.min())
        #jax.debug.print("dmax : {}", d.max())
        
        #jax.debug.print("min slice: {}", (center - density_plane_width / 2))
        #jax.debug.print("max slice: {}", (center + density_plane_width / 2))
        weight = jnp.where((d > (center - density_plane_width / 2)) & (d <= (center + density_plane_width / 2)), 1., 0.) # 243.40892, w=25
        #jax.debug.breakpoint(num_frames=1)
        # Painting density plane
        #jax.debug.print("boxsize x: {}", box_size[1])
        #jax.debug.print("max x: {}", jnp.max(x))
        #jax.debug.print("dplane x: {}",density_plane_npix[0] )
        
        #jax.debug.print("max x: {}", jnp.max(x))
        #jax.debug.print("boxsize y: {}", box_size[2])
        #jax.debug.print("max y: {}", jnp.max(y))
        #jax.debug.print("dplane y: {}",density_plane_npix[1] )
        
        density_plane = cic_paint_2d(jnp.zeros([density_plane_npix[0], density_plane_npix[1]]), jnp.c_[x,y], weight)

        #jax.debug.print("TOTPART : {}", jnp.sum(density_plane))
        # Apply density normalization
        #density_plane = density_plane / ( (box_size[1] / density_plane_npix[0]) * (box_size[2] / density_plane_npix[1]) *  w  )
        #jax.debug.print("norm x: {}", (box_size[1] / density_plane_npix[0]))
        #jax.debug.print("norm y: {}", (box_size[2] / density_plane_npix[1]))
        #jax.debug.print('norm d: {}', density_plane_width / box_size[0] * 2111  )
        #density_plane = density_plane / ( (box_size[1] / density_plane_npix[0]) * (box_size[2] / density_plane_npix[1]) *  w  )
        #density_plane =  density_plane*(box_size[0]/cubegrid_size)*(box_size[1]/cubegrid_size)*(box_size[2]/cubegrid_size)
        #density_plane = density_plane / ((nx / 256) * (ny / 256) * w)

        #density_plane = density_plane / ((gx / gx) *(gy / gy) * w)
        #w = density_plane_width / box_size[0] * gz


        density_plane = density_plane / (cubegrid_size**3/(density_plane_npix[0]*density_plane_npix[1]) * density_plane_width / box_size[0] )
        #jax.debug.print("norm: {}", (cubegrid_size**3/(density_plane_npix[0]*density_plane_npix[1]) * density_plane_width / box_size[0] ) )
        
        return density_plane
        
        
        #return y#positions
        
    #@jax.jit
    def neural_nbody_ode(a, state, args):
        """
        state is a tuple (position, velocities ) in internal units: [grid units, v=\frac{a^2}{H_0}\dot{x}]
        See this link for conversion rules: https://github.com/fastpm/fastpm#units
        """
        cosmo, params, _, _,_ = args
        pos = state[0]
        vel = state[1]

        kvec = fftk([cubegrid_size,cubegrid_size,cubegrid_size])
        #jax.debug.print("-------cubegridsize: {}", cubegrid_size)

        delta = cic_paint(jnp.zeros([cubegrid_size,cubegrid_size,cubegrid_size]), pos)

        delta_k = jnp.fft.rfftn(delta)

        # Computes gravitational potential
        pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

        # Apply a correction filter
        if params is not None:
            kk = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
            pot_k = pot_k * (1. + model.apply(params, kk, jnp.atleast_1d(a)))

        # Computes gravitational forces
        forces = jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), pos) for i in range(3)], axis=-1)
        forces = forces * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.stack([dpos, dvel], axis=0)



    def forward_model(cosmo, nz_shear, lin_field):
 
        # Create particles
        particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in [cubegrid_size,cubegrid_size,cubegrid_size]]),axis=-1).reshape([-1,3])

        cosmo = jc.Cosmology(Omega_c=cosmo.Omega_c, sigma8=cosmo.sigma8, Omega_b=cosmo.Omega_b, h=cosmo.h, n_s=cosmo.n_s, w0=cosmo.w0, Omega_k=0., wa=0.)

        # Temporary fix
        cosmo._workspace = {}

        # Initial displacement
        if method=='lpt':
            bs=[2070.6279240848658, 187.08286933869707, 110.67971810589329]
            particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in gs]),axis=-1).reshape([-1,3])
            #box_size  = [jnp.int32(jnp.ceil(box_size[1])),jnp.int32(jnp.ceil(box_size[2])),jnp.int32(jnp.ceil(box_size[0]))]
            #garid_size =  [187,110,2070] #[jnp.int32(jnp.ceil(box_size[1]))//2,jnp.int32(jnp.ceil(box_size[2]))//2,jnp.int32(jnp.ceil(box_size[0]))//2]
            # Compute the scale factor that corresponds to each slice of the volume
            r_center = (jnp.arange(gs[0]) + 0.5)*bs[0]/gs[0]
            a_center = jc.background.a_of_chi(cosmo, r_center)
            a        = a_center

            # Compute displacement and paint positions of particles onto lightcone
            eps,_     = lpt_lightcone(cosmo, lin_field, particles, a_center, [gs[1],gs[2],gs[0]])
            lightcone = cic_paint(jnp.zeros(gs),  particles+eps) 
            #eps, _    = lpt_lightcone(cosmo, lin_field, particles, a_center, [cubegrid_size,cubegrid_size,cubegrid_size])
            #lightcone = cic_paint(jnp.zeros(grid_size),  particles+eps) 

            # Apply de-cic filter to recover more signal on small scales
            dplanes = compensate_cic(lightcone)
            #jax.debug.breakpoint()

            dx = box_size[1]  / gs[1]
            dy = box_size[2]  / gs[2]
            dz = box_size[0]  / gs[0]


        elif method=='pm':

            assert density_plane_width is not None
            assert density_plane_npix is not None

            #density_plane_smoothing = 0.0001
            
            a_init    = 0.01
            n_lens    = int(box_size[0] // density_plane_width)
            r         = jnp.linspace(0., box_size[0], n_lens + 1)
            r_center  = 0.5 * (r[1:] + r[:-1])
            a_center  = jc.background.a_of_chi(cosmo, r_center)
            cc        = jc.background.radial_comoving_distance(cosmo, jnp.linspace(0.005,1,500)) 

            eps, p    = lpt_lightcone(cosmo, lin_field, particles, a_init, [cubegrid_size,cubegrid_size,cubegrid_size])
            term      = ODETerm(neural_nbody_ode)
            solver    = LeapfrogMidpoint()#Tsit5()#Tsit5()#KenCarp5()#Tsit5()#Heun()#LeapfrogMidpoint()#Dopri8()
            saveat    = SaveAt(ts=a_center[::-1], fn=density_plane_fn)
            
            solution  = diffeqsolve(term, solver, t0=a_init, t1=1., dt0=0.05,
                                    y0        = jnp.stack([particles+eps, p], axis=0),
                                    args      = (cosmo, None, density_plane_width, density_plane_npix, cc ),
                                    saveat    = saveat,
                                    adjoint   = diffrax.RecursiveCheckpointAdjoint(3),
                                    max_steps = 32)

            #solution  = diffeqsolve(term, solver, t0=a_init, t1=1., dt0=0.02,
            #                        y0        = jnp.stack([particles+eps, p], axis=0),
            #                        args      = (cosmo, None, density_plane_width, density_plane_npix, cc ),
            #                        saveat    = saveat,
            #                        adjoint   = diffrax.RecursiveCheckpointAdjoint(15),
            #                        max_steps = 50)

            # .RecursiveCheckpointAdjoint controls the granularity of checkpointing.
            # A lower value for substeps means more frequent checkpoints (less recomputation but higher memory usage).
            # A higher value means less frequent checkpoints (more recomputation but lower memory usage).


            dx = box_size[1] / density_plane_npix[0]
            dy = box_size[2] / density_plane_npix[1]
            #jax.debug.breakpoint()
            dz = density_plane_width

            dplanes = solution.ys#jax.vmap(lambda x: gaussian_smoothing(x, density_plane_smoothing / dx ))(solution.ys)

            
            dplanes   = dplanes[::-1]
            a         = solution.ts[::-1]
            dplanes = jnp.transpose(dplanes,axes=(1, 2, 0))
            
        # Defining the coordinate grid for lensing map
        # fieldsize == angular size of images
        # grid_size == output grid size
        xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, field_size[0], density_plane_npix[0], endpoint=False), # range of X coordinates
                                    jnp.linspace(0, field_size[1], density_plane_npix[1], endpoint=False)) # range of Y coordinates
        #jax.debug.breakpoint()

        #coords       = jnp.array((jnp.stack([xgrid, ygrid], axis=0)*u.deg).to(u.rad))
        coords = jnp.array((jnp.stack([xgrid, ygrid], axis=0))*0.017453292519943295 ) # deg->rad

        # Generate convergence maps by integrating over nz and source planes
        convergence_maps = [simps(lambda z: nz(z).reshape([-1,1,1]) * convergence_Born(cosmo, dplanes, r_center, a, dx, dz, coords, z), 0.01,1.4, N=50) for nz in nz_shear]
        #convergence_maps   = [np.zeros((10,10)) for nz in nz_shear]
        

        '''
        sub_x  = (density_plane_npix[0] // 4) * 4
        sub_y  = (density_plane_npix[1] // 4) * 4

        nx_out = jnp.int32(sub_x/4)
        ny_out = jnp.int32(sub_y/4)
        

        start_x = (density_plane_npix[0] - sub_x) // 2
        start_y = (density_plane_npix[1] - sub_y) // 2

        kmap_cut = [kmap[start_x:start_x + sub_x, start_y:start_y + sub_y]  for kmap in convergence_maps]
        '''

        #convergence_maps = jnp.transpose(convergence_maps,axes=(1, 0))
        
        #jax.debug.print("Npix x in: {}", convergence_maps[0].shape[0] )
        #jax.debug.print("Npix y in: {}", convergence_maps[0].shape[1] )
        

        #jax.debug.print("shapex in: {}", convergence_maps[0].shape[0] )
        #jax.debug.print("shapey in: {}", convergence_maps[0].shape[1] )
        #jax.debug.print("shapex out: {}", out_dim[1] )
        #jax.debug.print("shapey out: {}", out_dim[0] )
        #jax.debug.print("resampx: {}", resample_factor[0] )
        #jax.debug.print("resampy: {}", resample_factor[1] )
        #jax.debug.breakpoint()
        
        # Reshape the maps to desired resoluton
        #resample_factor  = (convergence_maps[0].shape[1] // int(box_size[1]), convergence_maps[0].shape[0] // int(box_size[2]))    
        resample_factor  = (convergence_maps[0].shape[0] // out_dim[1], convergence_maps[0].shape[1] // out_dim[0])
        convergence_maps = [kmap.reshape([out_dim[1], resample_factor[0],  out_dim[0], resample_factor[1] ]).mean(axis=1).mean(axis=-1) for kmap in convergence_maps]

        #jax.debug.print("Npix x in: {}", convergence_maps[0].shape[0] )
        #jax.debug.print("Npix y in: {}", convergence_maps[0].shape[1] )
        
        #jax.debug.breakpoint()
        jax.clear_caches()
        return convergence_maps#, dplanes
        #return dplanes

    return forward_model

#nx=jnp.int16(jnp.ceil(box_size[1]))
#ny=jnp.int16(jnp.ceil(box_size[2]))
#print(nx,ny)

#jnp.ceil(box_size[2])
#jnp.ceil(box_size[2])
#jax.debug.breakpoint(num_frames=1)

def even(X):
    Y=jnp.int32(jnp.round(X))
    if Y % 2 == 0:
        return Y + 2
    else:
        return Y + 1



dimx=even(box_size[1])
dimy=even(box_size[2])
#print('----',dimx,dimy)
#sys.exit()

# Generate the forward model given these survey settings
lensing_model = jax.jit(make_full_field_model( field_size = field_size,
                                               #field_npix = output_npix,
                                               cube_size   = cube_size,
                                               cubegrid_size  = cubegrid_size,
                                               box_size   = [float(box_size[0]),float(box_size[1]),float(box_size[2]) ],
                                               #grid_size  = grid_size,
                                               method     = method,
                                               density_plane_width = density_plane_width,
                                               #lens_sample_factor=1,
                                               density_plane_npix  = [int(dimx*3),int(dimy*3)],
                                               out_dim             = [int(dimx/2),int(dimy/2)],
                                               transform=transform,
                                               gs=[int(cubegrid_sizeT[0]),int(cubegrid_sizeT[1]),int(cubegrid_sizeT[2]) ]
                                             ))

def model():
    """
    This function defines the top-level forward model for our observations
    """
    Omega_b = 0.0492 
    Omega_c = numpyro.sample("omega_c", dist.Uniform(0.01, 1.5))
    #sigma8  = numpyro.sample("sigma8" , dist.Uniform(0.1, 2.0))
    S8      = numpyro.sample("S8" , dist.Uniform(0.1, 2.0))
    h       = 0.6726 
    w0      = -1     
    n_s     = 0.9645 

    Omega_m = Omega_b + Omega_c

    sigma8  = S8/(Omega_m/0.3)**0.5

    cosmo   = jc.Cosmology(Omega_c = Omega_c,
                            sigma8  = sigma8,
                            Omega_b = Omega_b,
                            Omega_k = 0.,
                            h   = h,
                            n_s = n_s,
                            w0  = w0,
                            wa  = 0.)

    numpyro.deterministic('sigma8', sigma8)
    numpyro.deterministic('Omega_m', Omega_m)
    
             
    
     
    # Create a small function to generate the matter power spectrum
    k     = jnp.logspace(-4, 1, 128)
    pk    = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    lin_field = linear_field([cubegrid_size,cubegrid_size,cubegrid_size], [cube_size,cube_size,cube_size], pk_fn)
    #jax.debug.breakpoint()


    # Generate random converge#nce map
    convergence_maps = lensing_model(cosmo, nz_shear, lin_field)
    #dplanes = lensing_model(cosmo, nz_shear, lin_field)
    
    numpyro.deterministic('kappa_noiseless_0', convergence_maps[0])
    numpyro.deterministic('kappa_noiseless_1', convergence_maps[1])
    numpyro.deterministic('kappa_noiseless_2', convergence_maps[2])

    #numpyro.deterministic('lin_field', lin_field)
    
    #numpyro.deterministic('dplane_0', dplanes[:,:,0])
    #numpyro.deterministic('dplane_1', dplanes[:,:,20])
    #numpyro.deterministic('dplane_2', dplanes[:,:,40])
    #numpyro.deterministic('dplane_3', dplanes[:,:,83])
    #numpyro.deterministic('dplanes', dplanes)
    

    #filtered_sigal = [fft.ifft2(fft.fft2(convergence_maps[i])*filt2d).real for i in range(len(convergence_maps)) ] 
  
    #observed_maps  = [numpyro.sample('kappa_obs_%d'%i, dist.MultivariateNormal(filtered_sigal[i].flatten(), scale_tril=scale_tril )) for i in range(len(convergence_maps)) ]

    #numpyro.deterministic('kappa_output_0', observed_maps[0])
    #numpyro.deterministic('kappa_output_1', observed_maps[1])
    #numpyro.deterministic('kappa_output_2', observed_maps[2])
    #return observed_maps

    #field_npix = convergence_maps[0]

    #jax.debug.print("fieldsizex: {}", field_size[0] )
    #jax.debug.print("outdimx: {}", out_dim[0] )
    #jax.debug.print("fieldsizey: {}", field_size[1] )
    #jax.debug.print("outdimy: {}", out_dim[1] )
        


    observed_maps = [numpyro.sample('kappa_%d'%i,dist.Normal(k, 0.26/jnp.sqrt(1*(field_size[0]*60/out_dim[0])*(field_size[1]*60/out_dim[1])  ) ))
                     for i,k in enumerate(convergence_maps)]

    numpyro.deterministic('kappa_output_0', observed_maps[0])
    numpyro.deterministic('kappa_output_1', observed_maps[1])
    numpyro.deterministic('kappa_output_2', observed_maps[2])

    return observed_maps


# Create a random realization of a map with fixed cosmology
gen_model     = condition(model, {"omega_c": 0.2664,
                                  "S8" : 0.8523,
                                   })

model_tracer = numpyro.handlers.trace(numpyro.handlers.seed(gen_model, jax.random.PRNGKey(0) ))
model_trace  = model_tracer.get_trace()

#np.save('/net/scratch/yomori//lin_field.npy',model_trace['lin_field']['value'])
#np.save('/net/scratch/yomori/dplane_0.npy',model_trace['dplane_0']['value'])
#np.save('/net/scratch/yomori/dplane_1.npy',model_trace['dplane_1']['value'])
#np.save('/net/scratch/yomori/dplane_2.npy',model_trace['dplane_2']['value'])
#np.save('/net/scratch/yomori/dplane_3.npy',model_trace['dplane_3']['value'])
#np.save('/net/scratch/yomori/dplanes.npy',model_trace['dplanes']['value'])

'''
np.save('/net/scratch/yomori/kappa_obs_0.npy',model_trace['kappa_output_0']['value'])
np.save('/net/scratch/yomori/kappa_obs_1.npy',model_trace['kappa_output_1']['value'])
np.save('/net/scratch/yomori/kappa_obs_2.npy',model_trace['kappa_output_2']['value'])
np.save('/net/scratch/yomori/kappa_noiseless_0.npy',model_trace['kappa_noiseless_0']['value'])
np.save('/net/scratch/yomori/kappa_noiseless_1.npy',model_trace['kappa_noiseless_1']['value'])
np.save('/net/scratch/yomori/kappa_noiseless_2.npy',model_trace['kappa_noiseless_2']['value'])
'''

jnp.savez(dir_out+'original_trace_%d_cubesize_%d_cubegridsize_%d_rebinned2.npz'%(run,cube_size,cubegrid_size),
                              kappa_1=model_trace['kappa_noiseless_0']['value'],
                              kappa_2=model_trace['kappa_noiseless_1']['value'],
                              kappa_3=model_trace['kappa_noiseless_2']['value'],
                              #kappa_4=model_trace['noiseless_convergence_3']['value'],
                              #kappa_5=model_trace['noiseless_convergence_4']['value'],
                              #field_size=field_size,
                              #output_npix=output_npix,
                              #z_mu    = z_mu,
                              #z_sigma = z_sigma
                              field_size_x=field_size[0],
                              field_size_y=field_size[1],
                              #output_npix_x=model_trace['noiseless_convergence_0']['value'].shape[0],
                              #output_npix_y=model_trace['noiseless_convergence_0']['value'].shape[1],
                              )



#Set "Fake data"
observed_model = condition(model, {'kappa_0': model_trace['kappa_0']['value'],
                                   'kappa_1': model_trace['kappa_1']['value'],
                                   'kappa_2': model_trace['kappa_2']['value'],
                                   })

nuts_kernel = numpyro.infer.NUTS(
                                model=observed_model,
                                init_strategy = partial(numpyro.infer.init_to_value, values={"omega_c": 0.2664,
                                                                                             "S8" : 0.8523,
                                                                                             'initial_conditions': model_trace['initial_conditions']['value']
                                                                                             #'omega_b': 0.0492,
                                                                                             #'h_0'    : 0.6727,
                                                                                             #'w_0'    : -1,
                                                                                            #'real_part': model_trace['real_part']['value'],
                                                                                            #'imag_part': model_trace['imag_part']['value'],
                                                                                             }),
                                max_tree_depth = 7,
                                step_size      = 1.0e-3
                                )


mcmc = numpyro.infer.MCMC(
                        nuts_kernel, 
                        num_warmup   = 0,
                        num_samples  = 70,
                        num_chains   = 1,
                        thinning     = 10,
                        progress_bar = True
                        )

if resume_state<0:

    print("---------------STARTING SAMPLING-------------------")
    mcmc.run( jax.random.PRNGKey(run))
    print("-----------------DONE SAMPLING---------------------")
    res = mcmc.get_samples()

    #summary(res, prob=0.9)
    # mean  : The mean of the posterior samples for each parameter.
    # std   : The standard deviation of the samples.
    # median: The median of the samples.
    # 5.0% and 95.0%: The 5th and 95th percentiles, representing a 90% credible interval by default (can be adjusted using the prob argument).
    # n_eff : The effective sample size, which reflects the number of independent samples.
    # R_hat : The potential scale reduction factor, which indicates how well the chains have converged (values close to 1.0 are good).
    
    #Save cosmo parameters
    np.save(dir_out+'cosmo_%s_%d_0.npy'%(name,run) ,np.c_[res['omega_c'],res['sigma8']])

    # Saving an intermediate checkpoint
    with open(dir_out+'%s_%d_0.pickle'%(name,run), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del res

    final_state = mcmc.last_state
    with open(dir_out+'/state_%s_%d_0.pkl'%(name,run), 'wb') as f:
        pickle.dump(final_state, f)

else:
    with open(dir_out+'state_%s_%d_%d.pkl'%(name,run,resume_state), 'rb') as f:
        mcmc.post_warmup_state = pickle.load(f)
    for i in range(resume_state+1,resume_state+2):
        mcmc.run(mcmc.post_warmup_state.rng_key)
        res = mcmc.get_samples()

        np.save(dir_out+'cosmo_%s_%d_%d.npy'%(name,run,i) ,np.c_[res['omega_c'],res['sigma8']])

        with open(dir_out+'/%s_%d_%d.pickle'%(name,run,i), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del res

        final_state = mcmc.last_state
        with open(dir_out+'/state_%s_%d_%d.pkl'%(name,run,i), 'wb') as f:
            pickle.dump(final_state, f)
