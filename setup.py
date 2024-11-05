from setuptools import find_packages, setup

long_description = open('README.md').read()

# Load requirements from requirements.txt
# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()
# Define dependencies in a list

requirements = [
    'gsl', 'numpy', 'scipy', 'swig', 'h5py', 
    'matplotlib', 'lalsuite', 'astropy', 'gwpy', 
    'pandas', 'numba', 'multiprocess', 'libconfig', 'python-dotenv', 
    'gwtools', 'gwsurrogate', 'wolframclient'
]

setup(name='gwModels',
      version='0.0.7',
      author='Tousif Islam',
      author_email='tousifislam24@gmail.com',
      license='MIT',
      include_package_data=True,
      contributors=['Tousif Islam'],
      description='Python package to provide gravitational waveform models',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=requirements,
      python_requires='>=3.10',  # Specify the required Python version here
      # will start new downloads if these are installed in a non-standard location
      # install_requires=["numpy","matplotlib","scipy"],
      classifiers=[
                'Intended Audience :: Other Audience',
                'Intended Audience :: Science/Research',
                'Natural Language :: English',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
      ],
)