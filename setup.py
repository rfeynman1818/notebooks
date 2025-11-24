from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).parent
with (HERE / "README.md").open(encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="driftlens",
    version="0.1.4",
    description="DriftLens: an Unsupervised Drift Detection framework",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    author="Salvatore Greco",
    author_email="grecosalvatore94@gmail.com",
    url="https://github.com/grecosalvatore/drift-lens",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.4",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.5.1,<3.6",
        "pandas>=1.1.3",
        "scipy>=1.10.0",
        "tqdm>=4.64.1",
        "setuptools>=58.0.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
