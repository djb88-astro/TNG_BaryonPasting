import setuptools

setuptools.setup(
    name="tng_baryonpasting_extraction",
    version="3.0.1",
    description="TNG halo properties calculations for Baryon Pasters collaboration",
    url="https://github.com/djb88-astro/TNG_BaryonPasting",
    author="David J Barnes",
    author_email="djbarnes@mit.edu",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy", "h5py", "astropy", "mpi4py", "numba"],
)
