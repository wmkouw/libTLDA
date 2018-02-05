from distutils.core import setup
from setuptools import find_packages
from os.path import join, dirname


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open(join(dirname(__file__), 'python/_version.py')) as versionpy:
    exec(versionpy.read())

with open('requirements.txt') as reqsfile:
    required = reqsfile.read().splitlines()

setup(
    name='libtlda',
    version='0.1.0',
    description=("Library of transfer learning and domain adaptation \
                  classifiers."),
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=required,
    url="https://github.com/wmkouw/libTLDA",
    license='Apache 2.0',
    author='Wouter Kouw',
    author_email='wmkouw@gmail.com',
    classifiers=['Topic :: Machine Learning :: classifiers',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7']
)
