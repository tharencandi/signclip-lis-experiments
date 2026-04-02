from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="signclip",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="SignCLIP: Sign Language-Text Contrastive Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/signclip-experiments",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pose-format>=0.9.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "omegaconf>=2.0",
        "tqdm>=4.60.0",
    ],
)
