from setuptools import find_packages, setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requiremnets
    '''
    HYPHEN_E='-e .'
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", " ") for req in requirements]
        
        if HYPHEN_E in requirements:
            requirements.remove(HYPHEN_E)
    return requirements
setup(
name='mlproject-1',
version='0.0.1',
author='Sriram',
author_email='sriram140606@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')


)