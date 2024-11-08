import healpy as hp
import os,sys
import numpy as np
import pyccl as ccl
import pylab as plt
import matplotlib.cm as cm
from scipy.special import erf
from scipy.stats import norm
from astropy.io import fits
import h5py
import pymaster as nmt
from scipy.ndimage import gaussian_filter as gf
import argparse

def get_theory(z_mu,z_sigma):
    import pyccl as ccl
    import numpy as np
    
    cosmo = ccl.Cosmology(Omega_c=0.2664, Omega_b=0.0492, h=0.6726, sigma8=0.831, n_s=0.9645)
    
    sigma_z  = z_sigma
    mu       = z_mu
    zz       = np.linspace(0, 2, 1000)
    nz1      = (1/(sigma_z*np.sqrt(2*np.pi)))*np.exp(-0.5*((zz-mu)/sigma_z)**2)
    
    z = np.linspace(0,4,1024)
    ks = np.logspace(-5,10,512)
    sf = (1./(1+z))[::-1]
    
    lpk_array = np.log(np.array([ccl.linear_matter_power(cosmo,ks,a) for a in sf]))
    
    pk = ccl.Pk2D(a_arr=sf, lk_arr=np.log(ks), pk_arr=lpk_array, is_logp=True)
    
    
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz1)) #CCL automatically normalizes dNdz
    
    #ell = np.linspace(10, 2000,25)
    ell = np.unique(np.geomspace(2, 2000, 64).astype(int))
    
    cls_lin1 = ccl.angular_cl(cosmo, lens1, lens1, ell, p_of_k_a=pk) #Cosmic shear
    
    
    lpk_array = np.log(np.array([ccl.nonlin_matter_power(cosmo,ks,a) for a in sf]))
    
    pk = ccl.Pk2D(a_arr=sf, lk_arr=np.log(ks), pk_arr=lpk_array, is_logp=True)
    
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz1)) #CCL automatically normalizes dNdz
    
    ell = np.linspace(10, 2000,25)
    cls_nlin1 = ccl.angular_cl(cosmo, lens1, lens1, ell, p_of_k_a=pk) #Cosmic shear
    
    cls_nlin = {1:cls_nlin1}#,2:cls_nlin2, 3:cls_nlin3, 4:cls_nlin4, 5:cls_nlin5}
    cls_lin  = {1:cls_lin1 }#,2:cls_lin2,  3:cls_lin3 , 4:cls_lin4 , 5:cls_nlin5}
    
    k_arr = np.geomspace(1E-4,1E1,256)
    a_arr = np.linspace(0.1,1,128)
    l_arr = np.unique(np.geomspace(2, 2000, 64).astype(int))
    
    hmd_200m = ccl.halos.MassDef200m
    cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)
    nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)
    bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
    pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)
    
    t_M    = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz1))
    
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
    pk_MMf = ccl.halos.halomod_Pk2D(cosmo, hmc, pM,
                                    lk_arr=np.log(k_arr), a_arr=a_arr)
    cl_MM = ccl.angular_cl(cosmo, t_M, t_M, l_arr, p_of_k_a=pk_MMf)
        
    return l_arr,cl_MM,cls_lin1




def pixwin(l, pixel_size_arcmin):
    pixel_size_rad = pixel_size_arcmin * (np.pi / (180.0 * 60.0))
    W_l = (np.sinc(l * pixel_size_rad / (2 * np.pi)))**2
    return W_l
'''
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
'''


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


def pspec(map1, map2, mask1, mask2, ell_edges, pix_size, N):
    """
    Calculates the power spectrum of a 2D map by FFTing, squaring, and azimuthally averaging.
    
    Parameters:
    - map1, map2: Input maps.
    - mask1, mask2: Masks applied to the maps.
    - ell_edges: Bin edges of ells for the power spectrum.
    - pix_size: Pixel size in arcminutes.
    - N: The number of pixels along one dimension of the map.
    
    Returns:
    - ells: The center of the ell bins.
    - cls: The power spectrum values for each ell bin.
    """
    
    N = int(N)
    # Make a 2D ell coordinate system
    ones  = np.ones(N)
    inds  = (np.arange(N) + 0.5 - N/2.) / (N - 1.)
    kX    = np.outer(ones, inds) / (pix_size / 60. * np.pi / 180.)
    kY    = np.transpose(kX)
    K     = np.sqrt(kX**2. + kY**2.)
    ell2d = K * 2. * np.pi

    # Get the 2D Fourier transform of the map
    F1    = np.fft.ifft2(np.fft.fftshift(map1 * mask1))
    F2    = np.fft.ifft2(np.fft.fftshift(map2 * mask2))
    PSD   = np.fft.fftshift(np.real(np.conj(F1) * F2))

    # Make an array to hold the power spectrum results
    N_bins = len(ell_edges) - 1
    ells   = np.zeros(N_bins)
    cls    = np.zeros(N_bins)

    # Compute the centers of the ell bins
    ells = 0.5 * (ell_edges[:-1] + ell_edges[1:])

    # Bin the power spectrum by the given ell edges
    for i in range(N_bins):
        mask = (ell2d >= ell_edges[i]) & (ell2d < ell_edges[i+1])
        cls[i] = np.sum(PSD * mask) / np.sum(mask)

    # Normalization
    nrm = np.mean(mask1 * mask2) ** 0.5 / (N * N) ** 0.5 / (pix_size / 60 / 180 * np.pi)

    return ells, cls / nrm**2 / pixwin(ells, pix_size)  # Assuming pixwin is defined elsewhere


parser = argparse.ArgumentParser()
parser.add_argument('seed'        , default=1 , type=int)
parser.add_argument('--lmax'      , nargs=1, type=int)
parser.add_argument('--noiseless' , default=False, dest='noiseless',action='store_true')
args = parser.parse_args()

seed = args.seed
lmax = (args.lmax)[0]
noiseless=args.noiseless


zz       = np.linspace(0, 1.5, 1000)
mu       = np.array([0.6,0.7,0.8]) #0.95
sigma_z  = np.array([0.05,0.05,0.05]) #0.025
ngal     = np.array([1.00,1.00,1.00])
sigma_e  = np.array([0.26,0.26,0.26])
nz       = [(1/(sigma_z[zi]*np.sqrt(2*np.pi)))*np.exp(-0.5*((zz-mu[zi])/sigma_z[zi])**2) for zi in range(3)] 
nz1 = nz[0]
nz2 = nz[1]
nz3 = nz[2]

'''
# Fiducial cosmology
cosmo = ccl.Cosmology(Omega_c=0.2664, Omega_b=0.0492, h=0.6727, sigma8=0.831, n_s=0.9645)
k_arr = np.geomspace(1E-4,1E1,256)
a_arr = np.linspace(0.1,1,128)
z_arr = np.linspace(0.1, 2.5, 500)

sigma_e   = 0.26

z  = np.linspace(0,4.0,1024)
ks = np.logspace(-5,2,512)
sf = (1./(1+z))[::-1]

#lpk_array = np.log(np.array([ccl.nonlinear_matter_power(cosmo,ks,a) for a in sf]))
lpk_array = np.log(np.array([ccl.nonlin_matter_power(cosmo,ks,a) for a in sf]))

pk = ccl.Pk2D(a_arr=sf, lk_arr=np.log(ks), pk_arr=lpk_array, is_logp=True)


lens1 = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz1)) #CCL automatically normalizes dNdz
lens2 = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz2)) #CCL automatically normalizes dNdz
lens3 = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz3)) #CCL automatically normalizes dNdz

l_arr=ell = np.arange(0, 6000)

cl_11 = ccl.angular_cl(cosmo, lens1, lens1, ell, p_of_k_a=pk) #Cosmic shear
cl_12 = ccl.angular_cl(cosmo, lens1, lens2, ell, p_of_k_a=pk) #Cosmic shear
cl_13 = ccl.angular_cl(cosmo, lens1, lens3, ell, p_of_k_a=pk) #Cosmic shear

cl_22 = ccl.angular_cl(cosmo, lens2, lens2, ell, p_of_k_a=pk) #Cosmic shear
cl_23 = ccl.angular_cl(cosmo, lens2, lens3, ell, p_of_k_a=pk) #Cosmic shear

cl_33 = ccl.angular_cl(cosmo, lens3, lens3, ell, p_of_k_a=pk) #Cosmic shear
'''

tls={}

for i in range(0,3):
    for j in range(0,3):

        import pyccl as ccl
        import numpy as np
        
        cosmo = ccl.Cosmology(Omega_c=0.2664, Omega_b=0.0492, h=0.6726, sigma8=0.831, n_s=0.9645)
        
        z = np.linspace(0,4,1024)
        ks = np.logspace(-5,10,512)
        sf = (1./(1+z))[::-1]
        
        
        
        #ell = np.linspace(10, 2000,25)
        #ell = np.unique(np.geomspace(2, 2000, 64).astype(int))
        
        #lpk_array = np.log(np.array([ccl.nonlin_matter_power(cosmo,ks,a) for a in sf]))
        #pk = ccl.Pk2D(a_arr=sf, lk_arr=np.log(ks), pk_arr=lpk_array, is_logp=True)
        
        #lens1 = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[i])) #CCL automatically normalizes dNdz
        #lens2 = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[j])) #CCL automatically normalizes dNdz
        
        ell = np.linspace(10, 2000,25)
        #cls_nlin1 = ccl.angular_cl(cosmo, lens1, lens2, ell, p_of_k_a=pk) #Cosmic shear
        
        k_arr = np.geomspace(1E-4,1E1,256)
        a_arr = np.linspace(0.1,1,128)
        l_arr = np.arange(1001)#np.unique(np.geomspace(2, 2000, 64).astype(int))
        
        hmd_200m = ccl.halos.MassDef200m
        cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)
        nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)
        bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
        pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)
        
        t_M1    = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[i]))
        t_M2    = ccl.WeakLensingTracer(cosmo, dndz=(zz, nz[j]))
        
        hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
        pk_MMf = ccl.halos.halomod_Pk2D(cosmo, hmc, pM,
                                        lk_arr=np.log(k_arr), a_arr=a_arr)
        cl_MM = ccl.angular_cl(cosmo, t_M1, t_M2, l_arr, p_of_k_a=pk_MMf)
        tls['%d-%d'%(i+1,j+1)] = cl_MM 
            
cl_11=tls['1-1']
cl_22=tls['2-2']
cl_33=tls['3-3']
cl_12=tls['1-2']
cl_23=tls['2-3']
cl_13=tls['1-3']

rng  =np.random.default_rng(seed)
alms = hp.synalm(np.c_[cl_11,cl_22,cl_33,cl_12,cl_23,cl_13].T,new=True)

if noiseless==False:
    nls = sigma_e[0]**2/(ngal[0]*3437.75**2)*np.ones_like(cl_11)
    n1 = hp.synalm(nls)
    alms[0]=alms[0]+n1

    nls = sigma_e[1]**2/(ngal[1]*3437.75**2)*np.ones_like(cl_11)
    n2 = hp.synalm(nls)
    alms[1]=alms[1]+n2
    
    nls = sigma_e[2]**2/(ngal[2]*3437.75**2)*np.ones_like(cl_11)
    n3 = hp.synalm(nls)
    alms[2]=alms[2]+n3



m1   = hp.alm2map(alms[0],4096)
m2   = hp.alm2map(alms[1],4096)
m3   = hp.alm2map(alms[2],4096)


#np.save('clsk.npy',hp.alm2cl(alms[2]) )
#sys.exit()
import reproject

def set_header(ra,dec,span,size=500):
    #Sets the header of output projection
    #span = angular dimensions project
    #size = size of the output image
    res = span/(size+0.0)*0.0166667
    return hdr

def h2f(hmap,target_header,coord_in='C'):
    #project healpix -> flatsky
    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='bilinear', nested=False)
    return pr


npix       = 76
field_size = 4.00 #deg
delta_ell  = 80

reso       = (field_size/npix)*60 # resolution in arcmin. 
print("reso:",reso)
hdr = fits.Header()
hdr.set('NAXIS'   , 2)
hdr.set('NAXIS1'  , npix) #rebinned =50pix + 10 +10 apod border
hdr.set('NAXIS2'  , npix)
hdr.set('CTYPE1'  , 'RA---ZEA')
hdr.set('CRPIX1'  , npix/2.0)
hdr.set('CRVAL1'  , 0.)
hdr.set('CDELT1'  , -reso/60.)# rebinned version
hdr.set('CUNIT1'  , 'deg')
hdr.set('CTYPE2'  , 'DEC--ZEA')
hdr.set('CRPIX2'  , npix/2.0)
hdr.set('CRVAL2'  , 0 )
hdr.set('CDELT2'  , reso/60.)# rebinned version
hdr.set('CUNIT2'  , 'deg')
hdr.set('COORDSYS','icrs')

#sh = 50,100//50,50,100//50

mpt={}
mpt[0]=h2f(m1,hdr,coord_in='C')#.reshape(sh).mean(-1).mean(1)
mpt[1]=h2f(m2,hdr,coord_in='C')#.reshape(sh).mean(-1).mean(1)
mpt[2]=h2f(m3,hdr,coord_in='C')#.reshape(sh).mean(-1).mean(1)

ang = 0.0166667*(reso)*npix  #angle of the fullfield in deg

print("angular extent: ",ang) 

# add noise in projected space
if noiseless==False:
    prefix=''
else:
    prefix='_noiseless'

Lx = ang * np.pi/180
Ly = ang * np.pi/180

#  - Nx and Ny: the number of pixels in the x and y dimensions
Nx = npix
Ny = npix

#l0_bins = np.arange(Nx/2) * 2 * np.pi/Lx
#lf_bins = (np.arange(Nx/2)+1) * 2 * np.pi/Lx
#b = nmt.NmtBinFlat(l0_bins, lf_bins)
#print( b.get_effective_ells())
#if lmax==400:
#    bine = np.linspace(80,lmax,7)
#elif lmax==800:
#    bine = np.linspace(80,lmax,15)
#else:
#    sys.exit("bad lmax")

#b    = nmt.NmtBinFlat(bine[:-1],bine[1:])
mask=np.ones((Nx,Ny))

#f  = {}
#f[0] = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt[0]])
#f[1] = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt[1]])
#f[2] = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt[2]])
#ells_uncoupled = b.get_effective_ells()

#cls={}

# The effective sampling rate for these bandpowers can be obtained calling:
#ells_uncoupled = b.get_effective_ells()
#print( b.get_effective_ells())
bine = np.arange(50,lmax+50,50)

cls={}
for i in range(0,3):
    for j in range(i,3):
        bell, cls['(%d,%d)'%(i,j)] = pspec(mpt[i], mpt[j], mask, mask, bine, reso, npix)

#cls['ells']=b.get_effective_ells()
from pathlib import Path
#import pdb; pdb.set_trace()
Path("cls_linear%s_lmax%d/"%(prefix,lmax) ).mkdir(parents=True, exist_ok=True)
np.savez('cls_linear%s_lmax%d/cls_seed%d.npz'%(prefix,lmax,seed)
                                                                , ells=bell #b.get_effective_ells()
                                                                , cl00=cls['(0,0)']#[0]
                                                                , cl01=cls['(0,1)']#[0]
                                                                , cl02=cls['(0,2)']#[0]
                                                                , cl11=cls['(1,1)']#[0]
                                                                , cl12=cls['(1,2)']#[0]
                                                                , cl22=cls['(2,2)']#[0]
                                                                )

#np.save('mpt0.npy',mpt[0])
#np.save('cl_33.npy',np.c_[l_arr,cl_33])

#print( b.get_effective_ells())
#np.save('testcls',cls['(%d,%d)'%(0,0)])
#print(cls['(0,0)'])
