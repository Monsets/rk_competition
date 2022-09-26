from setuptools import find_packages, setup

setup(
    name='rkcompetition',
    version='1.0',
    description='ds competition module',
    author='Monset',
    author_email='monset008@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'}
)