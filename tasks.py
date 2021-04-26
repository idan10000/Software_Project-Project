
from invoke import task


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task()
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task()
def run(c, n, k, Random=True):
    print("generating shared object files")
    build(c)
    c.run("python3.8.5 main.py {n:s} {k:s} {random:}".format(n=n, k=k, random=Random))


