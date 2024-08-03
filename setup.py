from setuptools import setup

setup(
    name='jax_lensing_fourier',
    version='1.0',
    description='A small package for comparing full-field lensing constraints using JAX',
    author='Eiffl, Justinezgh',
    packages=['jax_lensing_fourier'],
    install_requires=[
        'jax-cosmo',
        'numpyro',
        #'lenstools',
        #'chainconsumer',
        #'cmasher',
        'JaxPM @ git+https://github.com/DifferentiableUniverseInitiative/JaxPM.git',
        #'sbi_lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git'
    ]
)


