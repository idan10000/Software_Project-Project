#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <Python.h>


static void algorithm(int MAX_ITER, double **obs, double **centroids, int *clusters, double **sums);


long K, N, d;

static int *kmeans(int MAX_ITER, double **obs, double **centroids, int *clusters) {
    int i;
    double **sums;

    /* declare array variables */
    sums = malloc(sizeof(double *) * K);
    assert(sums != NULL);

    for (i = 0; i < K; ++i) {
        sums[i] = malloc(sizeof(double) * d);
        assert(sums[i] != NULL);
    }


    algorithm(MAX_ITER, obs, centroids, clusters, sums);


    /* free memory */
    for (i = K - 1; i >= 0; --i) {
        free(sums[i]);
    }
    free(sums);

    return clusters;

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
    tempCentroid = malloc(d * sizeof(double));
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
            if (!changedAny && tempCentroid[j] != centroids[i][j])
                changedAny = 1;
            centroids[i][j] = tempCentroid[j];
        }
    }
    free(clusterSizes);
    free(tempCentroid);
    return changedAny;
}


static void algorithm(int MAX_ITER, double **obs, double **centroids, int *clusters, double **sums) {
    char changedCluster;
    int i;
    changedCluster = 1;

    for (i = 0; i < MAX_ITER && changedCluster; ++i) {
        assignAllObservations(obs, centroids, clusters);
        changedCluster = updateCentroids(obs, centroids, clusters, sums);
    }

}

struct PyInput {
    int MAX_ITER;
    Py_ssize_t ln;
    Py_ssize_t lk;
    Py_ssize_t ld;
    PyObject *obsRow, *centRow, *item, *_lstObs, *_lstCent;
};

struct CInput {
    double **obs;
    double **centroids;
};

struct PyOutput {
    PyObject *python_cluster_list,
            *python_point_list,
            *python_int;
};

struct COutput {
    int **clusters;
    int *clusterAssignments;
    int *clusterAmounts;
    int *indexes;
};

void initInput(struct CInput *cInput) {
    int i;
    cInput->obs = malloc(sizeof(double *) * N);
    assert(cInput->obs != NULL);
    for (i = 0; i < N; ++i) {
        (cInput->obs)[i] = malloc(sizeof(double) * d);
        assert(cInput->obs[i] != NULL);
    }
    (cInput->centroids) = malloc(sizeof(double *) * K);
    assert((cInput->centroids) != NULL);

    for (i = 0; i < K; ++i) {
        (cInput->centroids)[i] = malloc(sizeof(double) * d);
        assert((cInput->centroids)[i] != NULL);
    }

}

void freeInput(struct CInput *cInput) {
    int i;

    for (i = N - 1; i >= 0; --i) {
        free((cInput->obs)[i]);
    }
    free((cInput->obs));

    for (i = K - 1; i >= 0; --i) {
        free((cInput->centroids)[i]);
    }
    free((cInput->centroids));
}

static void HandleInput(struct PyInput *input, struct CInput *cInput) {
    Py_ssize_t i, j;
    N = (long) PyObject_Length(input->_lstObs);
    K = (long) PyObject_Length(input->_lstCent);
    d = (long) PyObject_Length(PyList_GetItem(input->_lstObs, 0));
    input->ln = PyList_Size(input->_lstObs);
    input->lk = PyList_Size(input->_lstCent);
    input->ld = PyList_Size(PyList_GetItem(input->_lstObs, 0));

    initInput(cInput);

    /* Go over each item of the list and reduce it */
    for (i = 0; i < input->lk; i++) {
        input->obsRow = PyList_GetItem(input->_lstObs, i);
        input->centRow = PyList_GetItem(input->_lstCent, i);
        for (j = 0; j < input->ld; j++) {
            input->item = PyList_GetItem(input->obsRow, j);
            (cInput->obs)[i][j] = PyFloat_AsDouble(input->item);
            input->item = PyList_GetItem(input->centRow, j);
            (cInput->centroids)[i][j] = PyFloat_AsDouble(input->item);
        }
    }

    for (i = input->lk; i < input->ln; i++) {
        input->obsRow = PyList_GetItem(input->_lstObs, i);
        for (j = 0; j < input->ld; j++) {
            input->item = PyList_GetItem(input->obsRow, j);
            (cInput->obs)[i][j] = PyFloat_AsDouble(input->item);
        }
    }

}

static void initOutput(struct COutput *COutput) {
    int i;
    (COutput->clusterAssignments) = malloc(sizeof(int) * N);
    if ((COutput->clusterAssignments) == NULL) {
        //TODO: handle malloc error
        printf("error");
    }

    (COutput->clusterAmounts) = calloc(K, sizeof(int));
    if ((COutput->clusterAssignments) == NULL) {
        //TODO: handle malloc error
    }

    COutput->clusters = malloc(sizeof(int *) * K);
    if (COutput->clusters == NULL) {
        //TODO: handle malloc error
    }

    for (i = 0; i < K; ++i) {
        (COutput->clusters)[i] = malloc(sizeof(int) * (COutput->clusterAmounts)[i]);
        if ((COutput->clusters)[i] == NULL) {
            //TODO: handle malloc error
        }
    }
    COutput->indexes = calloc(K, sizeof(int));
}

static void freeOutput(struct COutput *cOutput) {
    free(cOutput->clusterAmounts);
    free(cOutput->indexes);
}

static void initPythonClustersList(struct PyOutput *output, struct COutput *COutput) {
    int i, j;
    PyObject *python_inner_list;

    for (i = 0; i < N; ++i)
        (COutput->clusterAmounts)[(COutput->clusterAssignments)[i]]++;

    for (i = 0; i < K; ++i)
        printf("%d,", COutput->clusterAmounts[i]);
    printf("\n\n");

    for (i = 0; i < N; ++i)
        printf("%d,", COutput->clusterAssignments[i]);
    printf("\n\n");

    for (i = 0; i < N; ++i) {

        j = (COutput->clusterAssignments)[i];

        (COutput->clusters)[j][(COutput->indexes)[j]] = i;

        (COutput->indexes)[j]++;

    }
        printf("1");

    output->python_cluster_list = PyList_New(K);
    for (i = 0; i < K; ++i) {
        python_inner_list = PyList_New((COutput->clusterAmounts)[i]);
        for (j = 0; j < (COutput->clusterAmounts)[i]; ++j) {
            output->python_int = Py_BuildValue("i", (COutput->clusters)[i][j]);
            PyList_SetItem(python_inner_list, j, output->python_int);
        }
        PyList_SetItem(output->python_cluster_list, i, python_inner_list);
    }
    printf("2");

}

static void HandleOutput(struct PyOutput *output, struct COutput *cOutput) {
    int i;

    (output->python_point_list) = PyList_New(N);

    for (i = 0; i < N; ++i) {
        output->python_int = Py_BuildValue("i", cOutput->clusterAssignments[i]);
        PyList_SetItem(output->python_point_list, i, output->python_int);
    }

    initPythonClustersList(output, cOutput);

    printf("3");
//    freeOutput(cOutput);
    printf("4\n");

}


static PyObject *kmeans_capi(PyObject *self, PyObject *args) {
    struct PyInput input;
    struct CInput cInput;
    struct PyOutput output;
    struct COutput cOutput;

    PyObject *python_tuple;

    /* init params from python */
    if (!PyArg_ParseTuple(args, "iOO", &input.MAX_ITER, &input._lstObs, &input._lstCent)) {
        return NULL;
    }

    if (!PyList_Check(input._lstObs) || !PyList_Check(input._lstCent))
        return NULL;


    printf("input\n");
    HandleInput(&input, &cInput);
    printf("%ld\n", K);
    printf("init output\n");

    initOutput(&cOutput);
    printf("kmeans\n");

    cOutput.clusterAssignments = kmeans(input.MAX_ITER, cInput.obs, cInput.centroids, cOutput.clusterAssignments);
    printf("output\n");

    HandleOutput(&output, &cOutput);
    printf("finished output\n");
    python_tuple = PyTuple_Pack(2, output.python_cluster_list, output.python_point_list);
    printf("finished tuples\n");
    freeInput(&cInput);

    return python_tuple;
}

static PyMethodDef _methods[] = {
        {"kmeans", (PyCFunction) kmeans_capi, METH_VARARGS, PyDoc_STR("Enter MAX_ITER + Observations + Centroids")},
        {NULL,     NULL,                      0,            NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        _methods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void) {
    PyObject *m;
    m = PyModule_Create(&_moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
