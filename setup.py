from setuptools import setup, find_packages

setup(
    name='sai2_environment',
    version='0.0.1',
    description='Python environment for Sai2-based robot learning benchmark',
    url='git@github.com:mwulmer/sai2_environment.git',
    author='Elie Aljalbout',
    author_email='elie.aljalbout@tum.de',
    license='unlicense',
    packages=find_packages(),
    zip_safe=False
)
