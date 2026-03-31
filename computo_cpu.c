/*
 * computo_cpu.c — operationes BLAS in CPU
 * ==============================================
 *
 * Functiones pfr_cpu_*_f et pfr_cpu_*_d — includuntur ab computo_plat.c
 * et computo_cuda.cu.  Non compilatur seorsum.
 * Parens plica debet includere computo.h et <stdlib.h>.
 */

/* ================================================================
 * GEMM: C = A * B (float)
 * ================================================================ */

int pfr_cpu_matmat_f(pfr_matrix_f_t *c,
                     const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    int m = a->m, k = a->n, n = b->n;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int l = 0; l < k; l++)
                s += a->data[i*k + l] * b->data[l*n + j];
            c->data[i*n + j] = s;
        }
    return 0;
}

/* ================================================================
 * GEMV: y = A * x (float)
 * ================================================================ */

int pfr_cpu_matvec_f(pfr_vector_f_t *y,
                     const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    int m = a->m, n = a->n;
    for (int i = 0; i < m; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++)
            s += a->data[i*n + j] * x->data[j];
        y->data[i] = s;
    }
    return 0;
}

/* ================================================================
 * DOT: *res = x . y (float)
 * ================================================================ */

int pfr_cpu_dotum_f(float *res,
                    const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    float s = 0.0f;
    for (int i = 0; i < x->n; i++)
        s += x->data[i] * y->data[i];
    *res = s;
    return 0;
}

/* ================================================================
 * SCAL: x = alpha * x (float)
 * ================================================================ */

int pfr_cpu_scalare_f(pfr_vector_f_t *x, float alpha)
{
    if (!x) return -1;
    for (int i = 0; i < x->n; i++)
        x->data[i] *= alpha;
    return 0;
}

/* ================================================================
 * AXPY: y = alpha * x + y (float)
 * ================================================================ */

int pfr_cpu_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    for (int i = 0; i < y->n; i++)
        y->data[i] += alpha * x->data[i];
    return 0;
}

/* ================================================================
 * GEMV^T: y = A^T * x (float)
 * ================================================================ */

int pfr_cpu_matvec_trans_f(pfr_vector_f_t *y,
                           const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->m != x->n || y->n != a->n) return -1;
    int m = a->m, n = a->n;
    for (int j = 0; j < n; j++) y->data[j] = 0.0f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            y->data[j] += a->data[i*n + j] * x->data[i];
    return 0;
}

/* ================================================================
 * GER: A += alpha * x * y^T (float)
 * ================================================================ */

int pfr_cpu_ger_f(pfr_matrix_f_t *a, float alpha,
                  const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!a || !x || !y) return -1;
    if (a->m != x->n || a->n != y->n) return -1;
    int m = a->m, n = a->n;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a->data[i*n + j] += alpha * x->data[i] * y->data[j];
    return 0;
}

/* ================================================================
 * GEMM: C = A * B (double)
 * ================================================================ */

int pfr_cpu_matmat_d(pfr_matrix_d_t *c,
                     const pfr_matrix_d_t *a, const pfr_matrix_d_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    int m = a->m, k = a->n, n = b->n;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int l = 0; l < k; l++)
                s += a->data[i*k + l] * b->data[l*n + j];
            c->data[i*n + j] = s;
        }
    return 0;
}

/* ================================================================
 * GEMV: y = A * x (double)
 * ================================================================ */

int pfr_cpu_matvec_d(pfr_vector_d_t *y,
                     const pfr_matrix_d_t *a, const pfr_vector_d_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    int m = a->m, n = a->n;
    for (int i = 0; i < m; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++)
            s += a->data[i*n + j] * x->data[j];
        y->data[i] = s;
    }
    return 0;
}

/* ================================================================
 * DOT: *res = x . y (double)
 * ================================================================ */

int pfr_cpu_dotum_d(double *res,
                    const pfr_vector_d_t *x, const pfr_vector_d_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    double s = 0.0;
    for (int i = 0; i < x->n; i++)
        s += x->data[i] * y->data[i];
    *res = s;
    return 0;
}

/* ================================================================
 * SCAL: x = alpha * x (double)
 * ================================================================ */

int pfr_cpu_scalare_d(pfr_vector_d_t *x, double alpha)
{
    if (!x) return -1;
    for (int i = 0; i < x->n; i++)
        x->data[i] *= alpha;
    return 0;
}

/* ================================================================
 * AXPY: y = alpha * x + y (double)
 * ================================================================ */

int pfr_cpu_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    for (int i = 0; i < y->n; i++)
        y->data[i] += alpha * x->data[i];
    return 0;
}
