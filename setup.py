"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as fobj:
    requirements = [lib.strip() for lib in fobj.readlines()]

test_requirements = []

setup(
    author="Nicolai Frosch",
    author_email="nicolaifrosch@dr-pfleger.de",
    python_requires='>=3.8',
    name="oee_prediction",
    version='1.0',
    packages=find_packages(include='modules'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description=("Tool to integrate data from Fastec and External sources, train models and frontend for OEE Prediction."),
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    zip_safe=False
)
