from setuptools import setup, find_packages

setup(
    name='icu',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    description='Who can be discharged from the Intensive Care Unit?',
    author_email='lorosanu@users.noreply.github.com',
    install_requires=[
        'numpy>=1.15.4',
        'scipy>=1.1.0',
        'pandas>=0.23.4',
        'matplotlib>=3.0.2',
        'scikit-learn>=0.20.0',
        'notebook>=5.7.2']
)
