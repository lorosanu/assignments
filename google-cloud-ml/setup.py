from setuptools import setup, find_packages

setup(
    name='mnist',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    description='MNIST model on Cloud ML Engine',
    author='Luiza Sarzyniec',
    author_email='lorosanu@users.noreply.github.com',
    license='MIT',
    install_requires=[
        'numpy',
        'tensorflow'],
    zip_safe=False)
