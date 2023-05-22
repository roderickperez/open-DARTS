from setuptools import setup, find_packages, Distribution
import os

# custom class to inform setuptools about self-comiled extensions in the distribution
# and hence enforce it to create platform wheel
class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

setup(
    # Add packages that are inside folder darts-package
    package_dir={'': "darts-package"},

    # Now only include already built libraries
    package_data={'darts': ['*.pyd', '*.so', '*.dll', 'html/*.*', 'docs/*.pdf', 'hdata/*.*', 'build_info.txt', 'whatsnew.txt']},

    # Package metadata
    description='Delft Advanced Research Terra Simulator',

    # Dependent packages (distributions)
    install_requires=['matplotlib', 'numpy', 'numba', 'scipy', 'pandas', 'meshio', 'gmsh', 'iapws', 
    'plotly', 'xlrd', 'pykrige', 'openpyxl'],

    classifiers=[
        'Programming Language :: Python :: 3',
	    'Programming Language :: C++',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    distclass=BinaryDistribution,
)
