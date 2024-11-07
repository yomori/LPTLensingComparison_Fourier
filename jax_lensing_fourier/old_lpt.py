import pickle
import os,sys
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.stats import norm
from numpyro.handlers import seed, trace, condition
import diffrax 
import argparse
from pathlib import Path
import numpy as np
from functools import partial
import jax.numpy.fft as fft
from jax.scipy.ndimage import map_coordinates
import scipy 
from jaxpm.pm import pm_forces, growth_factor, growth_rate
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint, compensate_cic, cic_paint_2d, cic_read
from jax_cosmo.scipy.integrate import simps

parser  = argparse.ArgumentParser()
parser.add_argument('method'      , default=None, type=str, help='')
parser.add_argument('run'         , default=None, type=int, help='')
parser.add_argument('resume_state', default=None, type=int, help='')
parser.add_argument('maxtreedepth', default=None, type=int, help='')
parser.add_argument('--butterworth', nargs=2, type=float, help='Third set of three numbers')
parser.add_argument('--trace'     , default=False, dest='trace',action='store_true')
parser.add_argument('--warmup'     , default=False, dest='warmup',action='store_true')
args         = parser.parse_args()
resume_state = args.resume_state
run          = args.run
method       = args.method
trace        = args.trace
warmup       = args.warmup
maxtreedepth = args.maxtreedepth
butterworth  = args.butterworth
 
assert method=='lpt'

def linear_field(mesh_shape, box_size, pk, field):
  """
    Generate initial conditions.
  """
  kvec = fftk(mesh_shape)
  kmesh = sum(
      (kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
      box_size[0] * box_size[1] * box_size[2])

  field = jnp.fft.rfftn(field) * pkmesh**0.5
  field = jnp.fft.irfftn(field)
  return field

def lpt_lightcone(cosmo, initial_conditions, positions, a, mesh_shape):
    """ Computes first order LPT displacement """
    initial_force = pm_forces(positions, delta=initial_conditions).reshape(mesh_shape+[3])
    a  = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a).reshape([1,1,-1,1]) * initial_force
    p  = (a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo, a)) * growth_factor(cosmo, a)).reshape([1,1,-1,1]) * initial_force
    return dx.reshape([-1,3]),p.reshape([-1,3])

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


def convergence_Born(cosmo,
                     density_planes,
                     r,
                     a,
                     dx,
                     dz,
                     coords,
                     z_source):
  """
  Compute the Born convergence
  Args:
    cosmo: `Cosmology`, cosmology object.
    density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.
  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """
  # Compute constant prefactor:
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
  # Compute comoving distance of source galaxies
  r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

  convergence = 0
  n_planes = len(r)
  
  def scan_fn(carry, i):
    density_planes, a, r = carry

    p = density_planes[:,:,i]
    density_normalization = dz * r[i] / a[i]
    p = (p - p.mean()) * constant_factor * density_normalization

    # Interpolate at the density plane coordinates
    im = map_coordinates(p, coords * r[i] / dx - 0.5, order=1, mode="wrap")

    return carry, im * jnp.clip(1. - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

  # Similar to for loops but using a jaxified approach
  _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

  return convergence.sum(axis=0)

def make_full_field_model(field_size, field_npix, box_shape, box_size, method='lpt',density_plane_width=None, density_plane_npix=None,density_plane_smoothing=None):
  

  def density_plane_fn(t, y, args):
    
    cosmo, _ , density_plane_width, density_plane_npix  = args
    positions = y[0]
    nx, ny, nz = box_shape

    # Converts time t to comoving distance in voxel coordinates
    w = density_plane_width / box_size[2] * box_shape[2]
    center = jc.background.radial_comoving_distance(cosmo, t) / box_size[2] * box_shape[2]

    xy = positions[..., :2]
    d = positions[..., 2]

    # Apply 2d periodic conditions
    xy = jnp.mod(xy, nx)

    # Rescaling positions to target grid
    xy = xy / nx * density_plane_npix

    # Selecting only particles that fall inside the volume of interest
    weight = jnp.where((d > (center - w / 2)) & (d <= (center + w / 2)), 1., 0.)

    # Painting density plane
    density_plane = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy, weight)

    # Apply density normalization
    density_plane = density_plane / ((nx / density_plane_npix) *
                                     (ny / density_plane_npix) * w)
    return density_plane
     
  @jax.jit
  def neural_nbody_ode(a, state, args):
    """
      state is a tuple (position, velocities ) in internal units: [grid units, v=\frac{a^2}{H_0}\dot{x}]
      See this link for conversion rules: https://github.com/fastpm/fastpm#units
      """
    cosmo, params, _, _ = args
    pos = state[0]
    vel = state[1]

    kvec = fftk(box_shape)

    delta = cic_paint(jnp.zeros(box_shape), pos)

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


  def forward_model(cosmo, nz_shear, initial_conditions):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    lin_field = linear_field(box_shape, box_size, pk_fn, initial_conditions)

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in box_shape]),axis=-1).reshape([-1,3])


    cosmo = jc.Cosmology(Omega_c=cosmo.Omega_c, sigma8=cosmo.sigma8, Omega_b=cosmo.Omega_b,
                        h=cosmo.h, n_s=cosmo.n_s, w0=cosmo.w0, Omega_k=0., wa=0.)
    # Temporary fix
    cosmo._workspace = {}
    
    # Initial displacement
    if method=='lpt':
      # Compute the scale factor that corresponds to each slice of the volume
      r_center = (jnp.arange(box_shape[-1]) + 0.5)*box_size[-1]/box_shape[-1]
      a_center = jc.background.a_of_chi(cosmo, r_center)
      a        = a_center

      # Compute displacement and paint positions of particles onto lightcone
      eps,_     = lpt_lightcone(cosmo, lin_field, particles, a_center, box_shape)
      lightcone = cic_paint(jnp.zeros(box_shape),  particles+eps) 
      
      # Apply de-cic filter to recover more signal on small scales
      lightcone = compensate_cic(lightcone)
      
      dx = box_size[0]  / box_shape[0]
      dz = box_size[-1] / box_shape[-1]


    elif method=='pm':

      assert density_plane_width is not None
      assert density_plane_npix is not None

      density_plane_smoothing = 0.1
      
      a_init    = 0.01
      n_lens    = int(box_size[-1] // density_plane_width)
      r         = jnp.linspace(0., box_size[-1], n_lens + 1)
      r_center  = 0.5 * (r[1:] + r[:-1])
      a_center  = jc.background.a_of_chi(cosmo, r_center)

      eps, p    = lpt_lightcone(cosmo, lin_field, particles, a_init, box_shape)
      term      = ODETerm(neural_nbody_ode)
      solver    = Dopri5()
      saveat    = SaveAt(ts=a_center[::-1], fn=density_plane_fn)
     
      solution  = diffeqsolve(term, solver, t0=0.01, t1=1., dt0=0.05,
                              y0        = jnp.stack([particles+eps, p], axis=0),
                              args      = (cosmo, neural_spline_params, density_plane_width, density_plane_npix),
                              saveat    = saveat,
                              adjoint   = diffrax.RecursiveCheckpointAdjoint(5),
                              max_steps = 32)

      dx = box_size[0] / density_plane_npix
      dz = density_plane_width

      lightcone = jax.vmap(lambda x: gaussian_smoothing(x, density_plane_smoothing / dx ))(solution.ys)
      lightcone = lightcone[::-1]
      a         = solution.ts[::-1]
      lightcone = jnp.transpose(lightcone,axes=(1, 2, 0))
    
 
    # Defining the coordinate grid for lensing map
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, field_size, box_shape[0], endpoint=False), # range of X coordinates
                                jnp.linspace(0, field_size, box_shape[1], endpoint=False)) # range of Y coordinates
    
    #coords       = jnp.array((jnp.stack([xgrid, ygrid], axis=0)*u.deg).to(u.rad))
    coords = jnp.array((jnp.stack([xgrid, ygrid], axis=0))*0.017453292519943295 ) # deg->rad

    # Generate convergence maps by integrating over nz and source planes
    convergence_maps = [simps(lambda z: nz(z).reshape([-1,1,1]) *
                              convergence_Born(cosmo, lightcone, r_center, a, dx, dz, coords, z), 0.01, 3.0, N=32) for nz in nz_shear]

    # Reshape the maps to desired resoluton
    convergence_maps = [kmap.reshape([field_npix, box_shape[0] // field_npix,  field_npix, box_shape[1] // field_npix ]).mean(axis=1).mean(axis=-1) for kmap in convergence_maps]

    return convergence_maps, lightcone

  return forward_model




dir  = '/net/scratch/yomori/sample_oldlpt/maxtreedepth%d/'%maxtreedepth
name = '%s_cubesettings_%d_%d'%(method,butterworth[0],butterworth[1])
Path(dir).mkdir(parents=True, exist_ok=True)


zz       = jnp.linspace(0, 1.5, 1000)
mu       = jnp.array([0.6,0.7,0.8]) #0.95
sigma_z  = jnp.array([0.05,0.05,0.05]) #0.025
ngal     = jnp.array([1.00,1.00,1.00])
sigma_e  = jnp.array([0.26,0.26,0.26])
nz       = [(1/(sigma_z[zi]*jnp.sqrt(2*jnp.pi)))*jnp.exp(-0.5*((zz-mu[zi])/sigma_z[zi])**2) for zi in range(3)] 
nz_shear = [jc.redshift.kde_nz(zz,nz[i],bw=zz[2]-zz[1], zmax=3.0, gals_per_arcmin2=ngal[i]) for i in range(3)]

nbins    = len(nz_shear)

Omega_b = 0.0492
Omega_c = 0.2664 
sigma8  = 0.831
h       = 0.6727
n_s     = 0.9645
w0      = -1
cosmo   = jc.Cosmology(Omega_c = Omega_c,
                       sigma8  = sigma8,
                       Omega_b = Omega_b,
                       Omega_k = 0.,
                       h   = h,
                       n_s = n_s,
                       w0  = w0,
                       wa  = 0.)


# Now, let's build a full field model
import numpyro
import numpyro.distributions as dist
#rom jax_lensing.model import make_full_field_model

# High resolution settings
# Note: this low resolution on the los impacts a tiny bit the cross-correlation signal,
# but it's probably worth it in terms of speed gains
#box_size  = [600., 600., 3500.]     # In Mpc/h [RUN2]
box_size  = [140.0, 140.0, 2011.0]   # In Mpc/h [RUN3]
box_shape = [76, 76, 256]            # Number of voxels/particles per side


# Specify the size and resolution of the patch to simulate
field_size = jnp.arctan2(box_size[0],box_size[-1])/jnp.pi*180.
field_npix = 76     # number of pixels per side
pixel_size = field_size * 60 / field_npix
print("Pixel size in arcmin: ", pixel_size)

reso    = (field_size/field_npix)*60 # resolution in arcmin. 
ang     = 0.0166667*(reso)*50  #angle of the fullfield in deg


# Noise covariance
print('Computing filter')
#f = filter(field_npix,ang/field_npix*0.0174533,lowpass) #low_pass_filter((50,50), 500, ang/npix*0.0174533)
filt2d = filter(reso,field_npix,butterworth[0],butterworth[1])

'''
ret      = np.zeros((int(field_npix*field_npix) ,50000))

for i in range(0,50000):
    tmp      = np.random.normal(np.zeros((field_npix ,field_npix )), sigma_e[0]/np.sqrt(ngal[0]*(reso)**2)) 
    y=fft.fft2(tmp)
    y=fft.fftshift(y)
    ret[:,i] = fft.ifft2(fft.fftshift(y*filt2d)).real.flatten()

noisecov = np.cov(ret)
np.save('old_lpt_noisecov_sige0.26_ngal1_%d_%d.npz.npy'%(butterworth[0],butterworth[1] ), noisecov )
sys.exit()
'''
noisecov   = np.load('old_lpt_noisecov_sige0.26_ngal1_%d_%d.npz.npy'%(butterworth[0],butterworth[1] ) )
scale_tril = scipy.linalg.cholesky(noisecov+np.eye(noisecov.shape[0])*1e-10 , lower=True)
del noisecov

# Generate the forward model given these survey settings
lensing_model = jax.jit(make_full_field_model( field_size  = field_size,
                                                field_npix = field_npix,
                                                box_size   = box_size,
                                                box_shape  = box_shape,
                                                method     = method,
                                                density_plane_width = 100,
                                                density_plane_npix  = 300
                                             ))
# Define the probabilistic model
def model():
    """
    This function defines the top-level forward model for our observations
    """
    # Sampling initial conditions
    initial_conditions = numpyro.sample('initial_conditions', dist.Normal(jnp.zeros(box_shape),
                                                                            jnp.ones(box_shape)))


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

    # Generate random convergence maps
    convergence_maps, _ = lensing_model(cosmo, nz_shear, initial_conditions)

    numpyro.deterministic('noiseless_convergence_0', convergence_maps[0])
    numpyro.deterministic('noiseless_convergence_1', convergence_maps[1])
    numpyro.deterministic('noiseless_convergence_2', convergence_maps[2])
    #numpyro.deterministic('noiseless_convergence_3', convergence_maps[3])
    #numpyro.deterministic('noiseless_convergence_4', convergence_maps[4])
    


    # Apply noise to the maps (this defines the likelihood)
    #observed_maps = [numpyro.sample('kappa_%d'%i,dist.Normal(k, sigma_e/jnp.sqrt(nz_shear[i].gals_per_arcmin2*(field_size*60/field_npix)**2)))for i,k in enumerate(convergence_maps)]
    #noise = [numpyro.sample('noise_%d'%i,dist.Normal(jnp.zeros_like(convergence_maps[0]), sigma_e/jnp.sqrt(nz_shear[i].gals_per_arcmin2*(field_size*60/field_npix)**2)))for i,k in enumerate(convergence_maps)]
    
    #filt = low_pass_filter(noise[0].shape,500,pixel_size)
    #import pdb;pdb.set_trace()
    #obslp = [fft.ifft2(fft.fft2(convergence_maps[i]+noise[i])*filt).real for i in range(len(convergence_maps)) ]
    #import pdb; pdb.set_trace()
    #f = filter(field_npix,ang/field_npix*0.0174533,lowpass) #low_pass_filter((50,50), 500, ang/npix*0.0174533)
    """
    reso_rad=ang/field_npix*0.0174533
    nsub = int(field_npix / 2 + 1)
    i, j = jnp.meshgrid(jnp.arange(nsub), jnp.arange(nsub))
    submatrix = 2 * jnp.pi * jnp.sqrt(i**2 + j**2) / reso_rad / jnp.float32(field_npix)

    result = jnp.zeros([field_npix, field_npix])
    result = result.at[0:nsub, 0:nsub].set(submatrix)
    result = result.at[0:nsub, nsub:].set(jnp.fliplr(submatrix[:, 1:-1]))
    result = result.at[nsub:, :].set(jnp.flipud(result[1:nsub-1, :]))
    tmp = jnp.around(result).astype(int)
    """
    obs_lp = []
    for i,k in enumerate(convergence_maps):
            #jax.debug.breakpoint(num_frames=1)
            obs_lp.append( fft.ifft2(fft.fftshift(fft.fftshift(fft.fft2(convergence_maps[i] ))*filt2d)).real)

    observed_maps = [numpyro.sample('kappa_%d'%i, dist.MultivariateNormal(obs_lp[i].flatten(), scale_tril=scale_tril)) for i in range(len(convergence_maps)) ]
        

    numpyro.deterministic('observed_maps_0', observed_maps[0])
    numpyro.deterministic('observed_maps_1', observed_maps[1])
    numpyro.deterministic('observed_maps_2', observed_maps[2])
    #numpyro.deterministic('observed_maps_3', observed_maps[1])
    #numpyro.deterministic('observed_maps_4', observed_maps[2])
    
    return observed_maps


# Create a random realization of a map with fixed cosmology
gen_model     = condition(model, {"omega_c": 0.2664,
                                  "S8"     : 0.8523,
                                   })

if trace is False:
    model_tracer = numpyro.handlers.trace(numpyro.handlers.seed(gen_model, jax.random.PRNGKey(0) ))
    model_trace  = model_tracer.get_trace()

    if run!=-1:
        np.savez(dir+'/model_trace_noiseless_convergence_mainseed_%s_%d.npz'%(method,run),conv1 = model_trace['noiseless_convergence_0']['value']
                                                                                         ,conv2 = model_trace['noiseless_convergence_1']['value']
                                                                                         ,conv3 = model_trace['noiseless_convergence_2']['value']
                                                                                         #,conv4 = model_trace['noiseless_convergence_3']['value']
                                                                                         #,conv5 = model_trace['noiseless_convergence_4']['value']
                                                                                         ,kappa1 = model_trace['kappa_0']['value']
                                                                                         ,kappa2 = model_trace['kappa_1']['value']
                                                                                         ,kappa3 = model_trace['kappa_2']['value']
                                                                                         #,kappa4 = model_trace['kappa_3']['value']
                                                                                         #,kappa5 = model_trace['kappa_4']['value']
                                                                                         #,convs3 = model_trace['observed_maps_2s']['value']
                                                                                         
                                                                                                            
                )
    

else:
    model_tracer = numpyro.handlers.trace(numpyro.handlers.seed(gen_model, run ))
    model_trace  = model_tracer.get_trace()

    np.savez(dir+'/model_trace_noiseless_convergence_differentseed_%s_%d.npz'%(method,run),kappa1 = model_trace['noiseless_convergence_0']['value']
                                                                                          ,kappa2 = model_trace['noiseless_convergence_1']['value']
                                                                                          ,kappa3 = model_trace['noiseless_convergence_2']['value']
                                                                                          #,kappa4 = model_trace['noiseless_convergence_3']['value']
                                                                                          #,kappa5 = model_trace['noiseless_convergence_4']['value']
                                                                                                        
            )
    sys.exit()


# Let's condition the model on the observed maps
observed_model = condition(model, {'kappa_0': model_trace['kappa_0']['value'],
                                   'kappa_1': model_trace['kappa_1']['value'],
                                   'kappa_2': model_trace['kappa_2']['value'],
                                   ##'kappa_3': model_trace['kappa_3']['value'],
                                   #'kappa_4': model_trace['kappa_4']['value'],
                                   })





nuts_kernel = numpyro.infer.NUTS(
                                model=observed_model,
                                init_strategy = partial(numpyro.infer.init_to_value, values={'omega_c': 0.2664,
                                                                                            "S8"     : 0.8523,
                                                                                            'initial_conditions': model_trace['initial_conditions']['value']}),
                                max_tree_depth = maxtreedepth,
                                step_size      = 4.0e-2,
                                #inverse_mass_matrix = mass_matrix
                            )

mcmc = numpyro.infer.MCMC(
                        nuts_kernel, 
                        num_warmup   = 0,
                        num_samples  = 500,
                        num_chains   = 1,
                        thinning     = 10,
                        progress_bar = True
                        )


if resume_state<0:

    print("---------------STARTING SAMPLING-------------------")
    mcmc.run( jax.random.PRNGKey(run))
    print("-----------------DONE SAMPLING---------------------")

    if warmup:
        mass_matrix = mcmc._last_state.adapt_state.inverse_mass_matrix
        with open(dir+'mass_matrix.pkl', 'wb') as f:
            pickle.dump(mass_matrix, f)
        sys.exit()

    res = mcmc.get_samples()

    np.save(dir+'cosmo_%s_%d_0.npy'%(name,run) ,np.c_[res['omega_c'],res['sigma8'],res['S8'] ])


    # Saving an intermediate checkpoint
    with open(dir+'%s_%d_0.pickle'%(name,run), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del res

    final_state = mcmc.last_state
    with open(dir+'/state_%s_%d_0.pkl'%(name,run), 'wb') as f:
        pickle.dump(final_state, f)

    #sys.exit()
    # Continue on
    for i in range(1,50):
        print('round',i,'done')
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key)
        res = mcmc.get_samples()
        np.save(dir+'cosmo_%s_%d_%d.npy'%(name,run,i),np.c_[res['omega_c'],res['sigma8'],res['S8'] ])

        with open(dir+'%s_%d_%d.pickle'%(name,run,i), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del res

        final_state = mcmc.last_state
        with open(dir+'/state_%s_%d_%d.pkl'%(name,run,i), 'wb') as f:
            pickle.dump(final_state, f)

else:
    # Save
    with open(dir+'state_%s_%d_%d.pkl'%(name,run,resume_state), 'rb') as f:
        mcmc.post_warmup_state = pickle.load(f)

    for i in range(resume_state+1,resume_state+50):
        mcmc.run(mcmc.post_warmup_state.rng_key)
        res = mcmc.get_samples()
        np.save(dir+'cosmo_%s_%d_%d.npy'%(name,run,i),np.c_[res['omega_c'],res['sigma8'],res['S8'] ])

        with open(dir+'/%s_%d_%d.pickle'%(name,run,i), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del res

        final_state = mcmc.last_state
        with open(dir+'/state_%s_%d_%d.pkl'%(name,run,i), 'wb') as f:
            pickle.dump(final_state, f)
