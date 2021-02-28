import argparse

from invoke import task


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task()
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")

@task()
def run(c, k, n, random):
    print("deleting shared object files")
    # c.run("rm *mykmeanssp*.so")
    print("generating shared object files")
    c.run("python3.8.5 setup.py build_ext --inplace")
    c.run("python3.8.5 main.py {n:s} {k:s} {random:s}".format(n=n, k=k, random=random))

@task()
def test(c, k, n, random):
    # print("deleting shared object files")
    # c.run("rm *mykmeanssp*.so")
    # print("generating shared object files")
    c.run("python setup.py build_ext --inplace")
    # c.run("python main.py {n:s} {k:s} {random:s}".format(n=n, k=k, random=random))