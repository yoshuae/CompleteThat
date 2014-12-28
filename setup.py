from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
      name='completethat',
      version='0.1dev',
      description='A package to solve low rank matrix completion problems',
      long_description=readme(),
      author='Joshua Edgerton, Esteban Fajardo',
      author_email='ef2451@columbia.edu, jae2154@columbia.edu',
      license='BSD',
      packages=['completethat'],
      install_requires=[
          'scipy', 'numpy', 'importlib'
      ],
      classifiers=[
         'Development Status :: 3 - Alpha',
         'License :: OSI Approved :: BSD License',
         'Programming Language :: Python :: 2.7',
         'Topic :: Scientific/Engineering :: Mathematics',
         'Topic :: Utilities'
      ],
      include_package_data=True,
      zip_safe=False
)

