import time

from invoke import task


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task()
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task()
def run(c, n, k, Random=True):
    # print("deleting shared object files")
    # c.run("rm *mykmeanssp*.so")
    print("generating shared object files")
    build(c)
    c.run("python3.8.5 main.py {n:s} {k:s} {random:}".format(n=n, k=k, random=Random))


@task()
def test(c, n, k, Random=True):
    # print("deleting shared object files")
    # c.run("rm *mykmeanssp*.so")
    # print("generating shared object files")
    c.run("python setup.py build_ext --inplace")
    c.run("python main.py {n:s} {k:s} {random:}".format(n=n, k=k, random=Random))


@task()
def timeTest(c, n, k, step):
    c.run("python3.8.5 setup.py build_ext --inplace")
    c.run("python3.8.5 timeTesting.py {n:s} {k:s} {step:}".format(n=n, k=k, step=step))
