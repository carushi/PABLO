from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PABLO',
    version='0.0.1',
    description="Publicly available brain image analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carushi/PABLO/",
    author="carushi",
    author_email="rkawaguc@cshl.edu",
    license="MIT",
    packages=find_packages(include=['PABLO', 'PABLO.*']),
    package_data={'PABLO': ['./atlas_data/all_harvard_mask.txt', './atlas_data/all_mni_mask.txt']},
    include_package_data=True
)    
