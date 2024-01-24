from setuptools import setup, find_packages
import os

def find_project_root():
    return os.path.abspath(os.path.dirname(__file__))

project_root = find_project_root()

setup(
    name='alexnet',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
)
