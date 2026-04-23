from setuptools import setup, find_packages

setup(
    name="iggt-reproduce",
    version="0.1.0",
    description="IGGT (Instance-Grounded Geometry Transformer) training reproduction based on VGGT",
    author="IGGT Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (see requirements.txt for full list)
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "hydra-core>=1.3",
        "numpy>=1.21",
        "opencv-python>=4.5",
    ],
    extras_require={
        "dev": [
            "black>=21.0",
            "pytest>=7.0",
        ]
    }
)
