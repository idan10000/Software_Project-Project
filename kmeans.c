#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <Python.h>


static char algorithm(int MAX_ITER, double **obs, double **centroids, int *clusters, double **sums);


long K, N, d;
int eps;

static char kmeans(int MAX_ITER, double **obs, double **centroids, int** clusters) {
    int i;
    char errorFlag;
    double **sums;


    /* declare array variables */
    sums = malloc(sizeof(double *) * K);
    if(sums == NULL){
        printf("Error allocating sums base array. Stacktrace:\n");
        PyErr_Print();
        PyErr_SetString(PyExc_MemoryError, "error allocating memory (malloc)");
        return 1;
    }

    for (i = 0; i < K; ++i) {
        sums[i] = malloc(sizeof(double) * d);
        if(sums[i] == NULL){
            printf("Error allocating sums inner array at index %d. Stacktrace:\n", i);
            PyErr_Print();
            PyErr_SetString(PyExc_MemoryError, "error allocating memory (malloc)");
            return 1;
        }
    }


    errorFlag = algorithm(MAX_ITER, obs, centroids, *clusters, sums);

    /* free memory */
    for (i = K - 1; i >= 0; --i) {
        free(sums[i]);
    }
    free(sums);
    if(errorFlag == 1)
        return 1;
    return 0;
}

static double norm(double *x, double *cluster) {
    double sum;
    int i;
    sum = 0;
    for (i = 0; i < d; ++i) {
        sum += (x[i] - cluster[i]) * (x[i] - cluster[i]);
    }
    return sum;
}

static int assignCluster(double *x, double **centroids) {
    double sum;
    double tempSum;
    int minCluster;
    int i;
    sum = norm(x, centroids[0]);
    minCluster = 0;

    for (i = 1; i < K; ++i) {
        tempSum = norm(x, centroids[i]);
        if (tempSum < sum) {
            sum = tempSum;
            minCluster = i;
        }
    }
    return minCluster;
}

static void assignAllObservations(double **obs, double **centroids, int *clusters) {
    int i;
    for (i = 0; i < N; ++i) {
        clusters[i] = assignCluster(obs[i], centroids);
    }
}

static void resetSums(double **sums) {
    int i, j;
    for (i = 0; i < K; ++i)
        for (j = 0; j < d; ++j)
            sums[i][j] = 0;
}

/**
 * after the clusters were updated, we update the centroids in a loop over the observations,
 * adding each observation coordinate to the corresponding centroids matrix coordinate.
 *
 * @param obs A matrix of the size N x d which represents the N d-dimensional vectors of the observations
 * @param centroids A matrix of the size K x d which represents the K d-dimensional vectors  of the centroids
 * @param clusters An array of the size N which represents the cluster that each observation is assigned to.
 * @param K The amount of clusters
 * @param N The amount of observations
 * @param d The size of the vector of each observation and centroid
 * @return If any of the centroids were updated
 */
static char updateCentroids(double **obs, double **centroids, int *clusters, double **sums) {
    int *clusterSizes;
    char changedAny;
    double *tempCentroid;
    int i, j;
    resetSums(sums);
    changedAny = 0;
    clusterSizes = calloc(K, sizeof(int));
    if(clusterSizes == NULL){
        printf("Error allocating clusterSizes array. Stacktrace:\n");
        PyErr_Print();
        PyErr_SetString(PyExc_MemoryError, "error allocating memory (calloc)");
        return 2;

    }

    tempCentroid = malloc(d * sizeof(double));
    if(tempCentroid == NULL){
        printf("Error allocating tempCentroid array. Stacktrace:\n");
        PyErr_Print();
        PyErr_SetString(PyExc_MemoryError, "error allocating memory (calloc)");
        return 2;

    }
    for (i = 0; i < N; ++i) {
        for (j = 0; j < d; ++j) {
            sums[clusters[i]][j] += obs[i][j];
        }
        clusterSizes[clusters[i]]++;
    }
    for (i = 0; i < K; ++i) {
        for (j = 0; j < d; ++j) {
            if (clusterSizes[i] != 0)
                tempCentroid[j] = sums[i][j] / clusterSizes[i];
            else
                tempCentroid[j] = centroids[i][j];
            if (!changedAny){
                if(tempCentroid[j] >= centroids[i][j]){
                    if(tempCentroid[j] - centroids[i][j] <= eps)
                        changedAny = 1;
                }
                else if(centroids[i][j] - tempCentroid[j]  <= eps)
                        changedAny = 1;
            }
            centroids[i][j] = tempCentroid[j];
        }
    }
    free(clusterSizes);
    free(tempCentroid);
    return changedAny;
}


static char algorithm(int MAX_ITER, double **obs, double **centroids, int *clusters, double **sums) {
    char changedCluster;
    int i;
    changedCluster = 1;

    for (i = 0; i < MAX_ITER && changedCluster; ++i) {
        assignAllObservations(obs, centroids, clusters);
        changedCluster = updateCentroids(obs, centroids, clusters, sums);
        if(changedCluster == 2)
            return 1;
    }
    return 0;

}

static PyObject* kmeans_capi(PyObject *self, PyObject *args)
{
    int MAX_ITER, *clusters;
    double **obs, **centroids;
    PyObject *obsRow, *centRow, *item, *_lstObs, *_lstCent; // INPUT variables

    PyObject *python_int, *python_cluster_list; // OUTPUT variables
    Py_ssize_t i, j, ln, lk, ld;

    /* init params from python */
    if(!PyArg_ParseTuple(args, "iOO", &MAX_ITER ,&_lstObs, &_lstCent)) {
        return NULL;
    }

    if (!PyList_Check(_lstObs) || !PyList_Check(_lstCent))
        return NULL;


    N = (long) PyObject_Length(_lstObs);
    K = (long) PyObject_Length(_lstCent);
    d = (long) PyObject_Length(PyList_GetItem(_lstObs,0));
    ln = PyList_Size(_lstObs);
    lk = PyList_Size(_lstCent);
    ld = PyList_Size(PyList_GetItem(_lstObs,0));
    eps = 0.0001;

    obs = malloc(sizeof(double *) * N);
    if(obs == NULL){
        printf("Error allocating obs base array. Stacktrace:\n");
        PyErr_Print();
        PyErr_SetString(PyExc_MemoryError, "error allocating memory (malloc)");
        Py_RETURN_NONE;
    }
    for (i = 0; i < N; ++i) {
        obs[i] = malloc(sizeof(double) * d);
        if(obs[i] == NULL){
            printf("Error allocating obs inner array. Stacktrace:\n");
            PyErr_Print();
            PyErr_SetString(PyExc_MemoryError, "error allocating memory (malloc)");
            Py_RETURN_NONE;
        }
    }
    centroids = malloc(sizeof(double *) * K);
    if(centroids == NULL){
        printf("Error allocating centroid base array. Stacktrace:\n");
        PyErr_Print();
        PyErr_SetString(PyExc_MemoryError, "error allocating memory (malloc)");
        Py_RETURN_NONE;
    }

    for (i = 0; i < K; ++i) {
        centroids[i] = malloc(sizeof(double) * d);
        if(centroids[i] == NULL){
            printf("Error allocating centroids inner array. Stacktrace:\n");
            PyErr_Print();
            PyErr_SetString(PyExc_MemoryError, "error allocating memory (malloc)");
            Py_RETURN_NONE;
        }
    }

    /* Go over each item of the list and reduce it */
    for (i = 0; i < lk; i++) {
        obsRow = PyList_GetItem(_lstObs, i);
        centRow = PyList_GetItem(_lstCent, i);
        for(j = 0; j < ld; j++){
            item = PyList_GetItem(obsRow, j);
            obs[i][j] = PyFloat_AsDouble(item);
            item = PyList_GetItem(centRow, j);
            centroids[i][j] = PyFloat_AsDouble(item);
        }
    }

    for (i = lk; i < ln; i++) {
        obsRow = PyList_GetItem(_lstObs, i);
        for(j = 0; j < ld; j++){
            item = PyList_GetItem(obsRow, j);
            obs[i][j] = PyFloat_AsDouble(item);
        }
    }

    clusters = malloc(N * sizeof(int));
    if(clusters == NULL){
        printf("Error allocating centroid base array. Stacktrace:\n");
        PyErr_Print();
        PyErr_SetString(PyExc_MemoryError, "error allocating memory (malloc)");
        Py_RETURN_NONE;
    }

    kmeans(MAX_ITER,obs,centroids, &clusters);

    python_cluster_list = PyList_New(N);
    for (i = 0; i < N; ++i){
        python_int = Py_BuildValue("i", clusters[i]);
        PyList_SetItem(python_cluster_list, i, python_int);
    }

    /* free memory */

    for (i = N - 1; i >= 0; --i) {
        free(obs[i]);
    }
    free(obs);
    for (i = K - 1; i >= 0; --i) {
        free(centroids[i]);
    }
    free(centroids);

    return python_cluster_list;
}

static PyMethodDef _methods[] = {
    {"kmeans", (PyCFunction)kmeans_capi, METH_VARARGS, PyDoc_STR("Enter MAX_ITER + Observations + Centroids")},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&_moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
