from distutils.core import setup
from setuptools import setup, find_packages
from os.path import join, dirname


def read(fname):
    """Read filename"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open(join(dirname(__file__), 'libtlda/_version.py')) as versionpy:
    exec(versionpy.read())

with open('requirements.txt') as reqsfile:
    required = reqsfile.read().splitlines()

setup(
    name='libtlda',
    version=__version__,
    description=("Library of transfer learning and domain adaptation \
                  classifiers."),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    keywords='machine learning, domain adaptation, transfer learning',
    packages=find_packages(),
    install_requires=required,
    url="https://github.com/wmkouw/libTLDA",
    license='Apache 2.0',
    author='Wouter Kouw',
    author_email='wmkouw@gmail.com',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries'],
    project_urls={'Tracker': 'https://github.com/wmkouw/libTLDA/issues'}
)
