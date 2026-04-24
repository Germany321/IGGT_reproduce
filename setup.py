from setuptools import setup, find_packages

setup(
    name="iggt-reproduce",
    version="0.1.0",
    description="IGGT (Instance-Grounded Geometry Transformer) training reproduction based on VGGT",
    author="IGGT Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "black>=21.0",
            "pytest>=7.0",
        ]
    }
)
