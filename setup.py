import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 7, 7):
    sys.exit('Hesaplamalı anlambilim first homework requires Python >= 3.7.7')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

with open('README.rst') as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name='odev_1',
    description='cs_hesaplamalı_anlambilim',
    long_description=readme,
    author='havvanur',
    author_email='drvshavva@gmail.com',
    packages=find_packages(exclude=['*tests*']),
    python_requires='==3.7.7',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
        ]
    },
)