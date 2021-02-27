from setuptools import setup, Extension, find_packages

setup(
    name='mykmeanssp',
    version='0.1',
    author='Idan & Yarden',
    install_requires=['invoke'],
    packages=find_packages(),  # find_packages(where='.', exclude=())
    #    Return a list of all Python packages found within directory 'where'
    license='GPL-2',

    ext_modules=[
        Extension(
            'mykmeanssp',
            ['kmeans.c'],
        )
    ]
)
