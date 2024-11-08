import healpy as hp
import os,sys
import numpy as np
import pyccl as ccl
import pylab as plt
import matplotlib.cm as cm
from scipy.special import erf
from scipy.stats import norm
from astropy.io import fits
from cobaya.run import run
from astropy.io import fits
from getdist import loadMCSamples

#import pymaster as nmt


def pixwin(l, pixel_size_arcmin):
    pixel_size_rad = pixel_size_arcmin * (np.pi / (180.0 * 60.0))
    W_l = (np.sinc(l * pixel_size_rad / (2 * np.pi)))**2
    return W_l

def pspec(map1, map2, mask1, mask2, delta_ell, ell_max, pix_size, N):
    "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"

    N = int(N)
    # make a 2d ell coordinate system
    ones  = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX    = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY    = np.transpose(kX)
    K     = np.sqrt(kX**2. + kY**2.)
    ell2d = K * 2. * np.pi

    # get the 2d fourier transform of the map
    F1    = np.fft.ifft2(np.fft.fftshift(map1*mask1))
    F2    = np.fft.ifft2(np.fft.fftshift(map2*mask2))
    PSD   = np.fft.fftshift(np.real(np.conj(F1) * F2))

    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ells   = np.zeros(N_bins)
    tmp    = np.arange(N_bins)
    cls    = np.zeros(N_bins)


    i_vals = np.arange(N_bins)[:, None]  # shape (N_bins, 1)
    masks = (tmp == i_vals).T  # shape (len(tmp), N_bins)
    updated_values = (i_vals + 0.5) * delta_ell
    ells = np.where(masks, updated_values, ells[:, None]).sum(axis=-1)

    i_vals = np.arange(N_bins)[:, None, None]
    lower_bounds = i_vals * delta_ell
    upper_bounds = (i_vals + 1) * delta_ell

    masks = (ell2d >= lower_bounds) & (ell2d < upper_bounds)  # Broadcasting should work now

    u_values = np.sum(PSD * masks, axis=(-1, -2)) / np.sum(masks, axis=(-1, -2))  # compute mean using masks

    cls = np.where(np.arange(N_bins)[:, None] == i_vals[:, 0, 0], u_values[:, None], cls).sum(axis=0)

    nrm = np.mean(mask1*mask2)**0.5 / (N*N)**0.5 / (pix_size/60/180*np.pi)

    return ells, cls/nrm**2/pixwin(ells, pix_size)#**2



def rebincl(ell,cl, bb):
    #bb   = np.linspace(minell,maxell,Nbins+1)
    Nbins=len(bb)-1
    ll   = (bb[:-1]).astype(np.int_)
    uu   = (bb[1:]).astype(np.int_)
    ret  = np.zeros(Nbins)
    retl = np.zeros(Nbins)
    err  = np.zeros(Nbins)
    for i in range(0,Nbins):
        ret[i]  = np.mean(cl[ll[i]:uu[i]])
        retl[i] = np.mean(ell[ll[i]:uu[i]])
        err[i]  = np.std(cl[ll[i]:uu[i]])
    return ret


lmax=int(sys.argv[1])

seed=1

bine = np.arange(50,lmax+50,50)
nbin = len(bine-1) 

#cosmo = ccl.Cosmology(Omega_c=0.2664, Omega_b=0.0492, h=0.6726, sigma8=0.831, n_s=0.9645)

S8=0.8523
omegam=0.2664+0.0492
sigma8  = S8/(omegam/0.3)**0.5
cosmo = ccl.Cosmology(Omega_c=omegam-0.0492, Omega_b=0.0492, h=0.6726, sigma8=sigma8, n_s=0.9645, w0=-1)


zz       = np.linspace(0, 1.5, 1000)
mu       = np.array([0.6,0.7,0.8]) #0.95
sigma_z  = np.array([0.05,0.05,0.05]) #0.025
ngal     = np.array([1.00,1.00,1.00])
sigma_e  = np.array([0.26,0.26,0.26])
nz       = [(1/(sigma_z[zi]*np.sqrt(2*np.pi)))*np.exp(-0.5*((zz-mu[zi])/sigma_z[zi])**2) for zi in range(3)] 
nz1 = nz[0]
nz2 = nz[1]
nz3 = nz[2]


#z   = np.linspace(0,4,1024)
#ks  = np.logspace(-5,10,512)
#sf  = (1./(1+z))[::-1]
#ell = np.linspace(10, 2000,25)
        
tls = {}

for i in range(0,3):
    for j in range(i,3):

        k_arr = np.geomspace(1E-4,1E1,256)
        a_arr = np.linspace(0.1,1,128)
        l_arr = np.arange(1001)#np.unique(np.geomspace(2, 2000, 64).astype(int))
        
        hmd_200m = ccl.halos.MassDef200m
        cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)
        nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)
        bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
        pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)
        
        t_M1   = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[i]))
        t_M2   = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[j]))
        
        hmc    = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
        pk_MMf = ccl.halos.halomod_Pk2D(cosmo, hmc, pM,
                                        lk_arr=np.log(k_arr), a_arr=a_arr)
        cl_MM  = ccl.angular_cl(cosmo, t_M1, t_M2, l_arr, p_of_k_a=pk_MMf)
        tls['%d-%d'%(i+1,j+1)] = cl_MM 
            
cl_11=tls['1-1']
cl_22=tls['2-2']
cl_33=tls['3-3']
cl_12=tls['1-2']
cl_23=tls['2-3']
cl_13=tls['1-3']


scl11 = rebincl(l_arr,cl_11,bine)
scl12 = rebincl(l_arr,cl_12,bine)
scl13 = rebincl(l_arr,cl_13,bine)

scl22 = rebincl(l_arr,cl_22,bine)
scl23 = rebincl(l_arr,cl_23,bine)

scl33 = rebincl(l_arr,cl_33,bine)

bcl   = np.concatenate([scl11,scl22,scl33,scl12,scl23,scl13])


#np.save('theory.npy',np.c_[l_arr,cl_11])
######################################################################################
npix       = 76
field_size = 4  #deg
reso       = (field_size/npix)*60 # resolution in arcmin. 
ang        = 0.0166667*(reso)*npix  #angle of the fullfield in deg


#ang = 0.0166667*9.73*70  #angle of the fullfield in deg

#Lx = ang * np.pi/180
#Ly = ang * np.pi/180
#  - Nx and Ny: the number of pixels in the x and y dimensions
#Nx = 76
#Ny = 76

#l0_bins = np.arange(Nx/2) * 2 * np.pi/Lx
#lf_bins = (np.arange(Nx/2)+1) * 2 * np.pi/Lx
#lm_bins = (np.arange(Nx/2)+0.5) * 2 * np.pi/Lx

#b = nmt.NmtBinFlat(l0_bins, lf_bins)
#print( b.get_effective_ells())

#import pdb; pdb.set_trace()
'''
# simulated dvec
bcl11 = rebincl(l_arr,cl_11,bine)
bcl12 = rebincl(l_arr,cl_12,bine)
bcl13 = rebincl(l_arr,cl_13,bine)
bcl22 = rebincl(l_arr,cl_22,bine)
bcl23 = rebincl(l_arr,cl_23,bine)
bcl33 = rebincl(l_arr,cl_33,bine)
bcl   = np.concatenate([bcl11,bcl22,bcl33,bcl12,bcl23,bcl13])
'''

y=np.load('/lcrc/project/SPT3G/users/ac.yomori/repo/sbi_lens/pspec_cube/cls_linear_noiseless_lmax%d/cls_seed%d.npz'%(lmax,1))

#cov
#cl = np.zeros(((len(bine)-1)*6,1000))
cl = np.zeros((len(y['ells'])*6,1000))

for i in range(1,1001):
    f=np.load('/lcrc/project/SPT3G/users/ac.yomori/repo/sbi_lens/pspec_cube/cls_linear_lmax%d/cls_seed%d.npz'%(lmax,i))
    cl11=f['cl00']
    cl12=f['cl01']
    cl13=f['cl02']
    cl22=f['cl11']
    cl23=f['cl12']
    cl33=f['cl22']
    
    cl[:,i-1]=np.concatenate([cl11,cl22,cl33,cl12,cl23,cl13])

cov = np.cov(cl)
#bcl=np.mean(cl,axis=1)
#np.save('bins.npy',lm_bins[3:11])
#np.save('dvec.npy',bcl)
#np.save('cov.npy',cov)
#sys.exit()

hartlap = (1000-2-len(bcl))/(1000-1)
icov    = np.linalg.pinv(cov)*hartlap

def dlike(omegam,S8):
    
    # Fiducial cosmology
    sigma8  = S8/(omegam/0.3)**0.5
    cosmo = ccl.Cosmology(Omega_c=omegam-0.0492, Omega_b=0.0492, h=0.6726, sigma8=sigma8, n_s=0.9645, w0=-1)

    #z   = np.linspace(0,4,1024)
    #ks  = np.logspace(-5,10,512)
    #sf  = (1./(1+z))[::-1]
    #ell = np.linspace(10, 2000,25)
         
    tls = {}

    for i in range(0,3):
        for j in range(i,3):

            k_arr = np.geomspace(1E-4,1E1,256)
            a_arr = np.linspace(0.1,1,128)
            l_arr = np.arange(1001)#np.unique(np.geomspace(2, 2000, 64).astype(int))
            
            hmd_200m = ccl.halos.MassDef200m
            cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)
            nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)
            bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
            pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)
            
            t_M1   = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[i]))
            t_M2   = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[j]))
            
            hmc    = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
            pk_MMf = ccl.halos.halomod_Pk2D(cosmo, hmc, pM,
                                            lk_arr=np.log(k_arr), a_arr=a_arr)
            cl_MM  = ccl.angular_cl(cosmo, t_M1, t_M2, l_arr, p_of_k_a=pk_MMf)
            tls['%d-%d'%(i+1,j+1)] = cl_MM 
                
    cl_11=tls['1-1']
    cl_22=tls['2-2']
    cl_33=tls['3-3']
    cl_12=tls['1-2']
    cl_23=tls['2-3']
    cl_13=tls['1-3']


    scl11 = rebincl(l_arr,cl_11,bine)
    scl12 = rebincl(l_arr,cl_12,bine)
    scl13 = rebincl(l_arr,cl_13,bine)
    
    scl22 = rebincl(l_arr,cl_22,bine)
    scl23 = rebincl(l_arr,cl_23,bine)

    scl33 = rebincl(l_arr,cl_33,bine)

    scl   = np.concatenate([scl11,scl22,scl33,scl12,scl23,scl13])

    X     = bcl-scl
    chi2  = X @ icov @ X
    return -0.5*chi2


def get_sigma8(omegam,S8):
    return S8/(omegam/0.3)**0.5

pars  = {
          'omegam'    : {'prior': {'min': 0.0492+0.01 , 'max': 0.0492+1.5}, 'latex': r'\Omega_{\rm m}'},
          'S8'        : {'prior': {'min': 0.1         , 'max': 2.0       }, 'latex': r'S_{8}'},
          'sigma8'    : {"derived": get_sigma8, 'latex':r"\sigma_8" }
        }

info =  {
          "params"    : pars,
          "likelihood": {'testlike': {"external": dlike } },
          "sampler"  : {"mcmc": {"max_samples": 100000, "Rminus1_stop": 0.05, "max_tries": 100000}},
          #"sampler"   : {"polychord": {"path":'/lcrc/project/SPT3G/users/ac.yomori/scratch/testcobaya3/PolyChordLite/',
          #                             "precision_criterion": 0.01,
          #                             'nlive': '20d',
          #                             'num_repeats': '5d'}},
          "output"    : 'chains_uniform_s8/chain_S8_%d.txt'%lmax,
          "force"     : True,
          'feedback'  : 9999999
        }


updated_info, products = run(info)















