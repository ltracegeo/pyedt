import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="pyedt",
    version="0.1.5",
    author="LTrace technologies",
    description="Euclidian Distance Transform functions for GPU and parallel CPU",
    packages=["pyedt"],
    url='https://pypi.org/project/pyedt/',
    install_requires=[
        'matplotlib>=3.5.1',
        'numba>=0.56.2',
        'numpy>=1.23.1',
        'scipy>=1.8.1',
        'pytest>=7.4.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    )
