"""
PNLP 

"""
from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

PYTHON_REQUIRES = '>=3.7.*'    
setup(
    name='pnlp',
    version='0.1',
    description='protien sequence NLP',
    long_description=long_description,
    url='',
    author='Bin Hu, Michal Babinski, Kaetlyn Gibson',
    author_email='{bhu, mbabinski, kaetlyn}@lanl.gov',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: BSD',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    keywords='Protein, natural language processing, NLP',
    packages=find_packages('src', exclude=['contrib', 'docs', 'tests']),
    package_dir={'': 'src'},
    # install_requires=['torch', 'pandas', 'click'],  # Optional


    # If there are data files included in your packages that need to be
    # installed, specify them here.

    # package_data={  # Optional
    #     'akimpute': ['data/rs_chrom_db_sorted.tsv', 'data/example.config'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],
    entry_points={  # Optional
        'console_scripts': [
            ''
        ],
    },
)
