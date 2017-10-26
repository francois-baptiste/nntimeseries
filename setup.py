from distutils.core import setup

setup(
    name='nnts',
    version='0.0.1',
    author='mbinkowski',
    author_email='',
    packages=['nnts'],
    scripts=[],
    license='LICENSE.txt',
    description='Useful towel-related stuff.',
    install_requires=[
        "keras",
        "numpy",
        "pandas",
        "h5py"
    ],
)
