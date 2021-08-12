import setuptools
import re

with open("yolox_backbone/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
setuptools.setup(
    name="yolox_backbone",
    version="0.0.1",
    license='Apache',
    python_requires=">=3.6",
    author="Yonghye Kwon",
    author_email="developer.0hye@gmail.com",
    description="yolox_backbone is a deep-learning library and is a collection of YOLOX Backbone models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/developer0hye/YOLOX-Backbone",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
    ],
)
