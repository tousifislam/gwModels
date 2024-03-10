from setuptools import find_packages, setup

long_description = open('README.md').read()

setup(name='gwModels',
      version='0.0.1',
      author='Tousif Islam',
      author_email='tousifislam24@gmail.com',
      #packages=['BHPTNRremnant'],
      #packages=find_packages(),
      packages=['gwModels'],
      license='MIT',
      include_package_data=True,
      contributors=['Tousif Islam'],
      description='Python package to provide gravitational waveform models',
      long_description=long_description,
      long_description_content_type='text/markdown',
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