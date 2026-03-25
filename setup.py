from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    requirements: List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            # Read lines from file
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip()
                ## ignore empty line and -e
                if requirement and requirement != '-e .':
                    requirements.append(requirement)
    except FileNotFoundError:
        print('requirements.txt file not found')

    return requirements

setup(
    name="Network Security",
    version="0.0.1",
    author="Jawahar",
    author_email="jeeva.r.anand6@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

