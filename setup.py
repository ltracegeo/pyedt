import setuptools

setuptools.setup(
    name="pyedt",
    version="0.1.3",
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
        'License :: MIT License',
    ],
    )
