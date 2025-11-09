from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name='station',
    version='1.0.0',
    packages=find_packages(), # Automatically find 'station' and 'station.rooms'
    install_requires=parse_requirements('requirements.txt'),
)