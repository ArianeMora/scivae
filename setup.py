from setuptools import setup
import os
import re


def read_version():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scivae/__init__.py')
    with open(path, 'r') as fh:
        return re.search(r'__version__\s?=\s?[\'"](.+)[\'"]', fh.read()).group(1)


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='scivae',
      version=read_version(),
      description='',
      long_description=readme(),
      long_description_content_type='text/markdown',
      author='Ariane Mora',
      author_email='ariane.n.mora@gmail.com',
      url='https://github.com/ArianeMora/scivae',
      license='GPL3',
      project_urls={
          "Bug Tracker": "https://github.com/ArianeMora/scivae/issues",
          "Documentation": "https://github.com/ArianeMora/scivae",
          "Source Code": "https://github.com/ArianeMora/scivae",
      },
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords='util',
      packages=['scivae'],
      entry_points={
          'console_scripts': [
              'scivae = scivae.__main__:main'
          ]
      },
      install_requires=['pandas', 'numpy', 'keras', 'matplotlib>=3.3.3', 'sciutil',
                        'pyswarm', 'sklearn', 'tensorflow',
                        'tensorflow-probability', 'seaborn', 'sciviso>=1.0.1'],
      python_requires='>=3.6',
      data_files=[("", ["LICENSE"])]
      )