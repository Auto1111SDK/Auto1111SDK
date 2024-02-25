from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.95'
DESCRIPTION = 'SDK for Automatic 1111.'
LONG_DESCRIPTION = 'A package that allows you to easily generate images and run diffusion models the same way as Automatic 1111.'

# Setting up
setup(
    name="auto1111sdk",
    version=VERSION,
    author="Auto1111 SDK",
    author_email="saketh.kotamraju@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
    "GitPython==3.1.32",
    "Pillow==9.5.0",
    "accelerate==0.21.0",
    "basicsr==1.4.2",
    "blendmodes==2022",
    "clean-fid==0.1.35",
    "einops==0.4.1",
    "fastapi==0.94.0",
    "gfpgan==1.3.8",
    "gradio==3.41.2",
    "httpcore==0.15",
    "inflection==0.5.1",
    "jsonmerge==1.8.0",
    "kornia==0.6.7",
    "lark==1.1.2",
    "numpy==1.23.5",
    "omegaconf==2.2.3",
    "open-clip-torch==2.20.0",
    "piexif==1.1.3",
    "psutil==5.9.5",
    "pytorch_lightning==1.9.4",
    "realesrgan==0.3.0",
    "resize-right==0.0.2",
    "safetensors==0.3.1",
    "scikit-image==0.21.0",
    "timm==0.9.2",
    "tomesd==0.1.3",
    "torch==2.1.0",
    "torchdiffeq==0.2.3",
    "torchsde==0.2.6",
    "transformers==4.30.2",
    "httpx==0.24.1",
    "clip"
],
    keywords=['python', 'Automatic 1111', 'Stable Diffusion Web UI', 'image generation', 'stable diffusion', 'civit ai'], 
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)