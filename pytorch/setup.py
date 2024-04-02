from setuptools import setup, find_packages

setup(
    name='mygpt',
    version='0.1.0',
    description='A Python package implementing classes for GPT models',
    author='Nicolas Agostini',
    author_email='n.b.agostini@gmail.com',
    packages=find_packages(where='mygpt'),
    package_dir={'': 'mygpt'},
    install_requires=[
        'torch',
        'torchvision',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
