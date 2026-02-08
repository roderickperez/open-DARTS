from setuptools import setup, Distribution

# custom class to inform setuptools about self-compiled extensions in the distribution
# and hence enforce it to create platform wheel
class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

setup(
    # Add package dartsflash (only)
    packages = ['dartsflash'],

    # Now only include already built libraries
    package_data={'dartsflash': ['*.pyd', '*.so', '*.dll', '*/*']},

    distclass=BinaryDistribution
)

