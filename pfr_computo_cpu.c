/*
 * pfr_computo_cpu.c — operationes BLAS in CPU
 * ==============================================
 *
 * Functiones pfr_cpu_* — includuntur ab pfr_computo_plat.c
 * et pfr_computo_cuda.cu.  Non compilatur seorsum.
 * Parens plica debet includere pfr_computo.h et <stdlib.h>.
 */

/* ================================================================
 * GEMM: C = A * B
 * ================================================================ */

int pfr_cpu_matmat(pfr_matrix_t *c,
                      const pfr_matrix_t *a, const pfr_matrix_t *b)
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
 * GEMV: y = A * x
 * ================================================================ */

int pfr_cpu_matvec(pfr_vector_t *y,
                      const pfr_matrix_t *a, const pfr_vector_t *x)
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
 * DOT: *res = x . y
 * ================================================================ */

int pfr_cpu_dotum(float *res,
                     const pfr_vector_t *x, const pfr_vector_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    float s = 0.0f;
    for (int i = 0; i < x->n; i++)
        s += x->data[i] * y->data[i];
    *res = s;
    return 0;
}

/* ================================================================
 * SCAL: x = alpha * x
 * ================================================================ */

int pfr_cpu_scalare(pfr_vector_t *x, float alpha)
{
    if (!x) return -1;
    for (int i = 0; i < x->n; i++)
        x->data[i] *= alpha;
    return 0;
}

/* ================================================================
 * AXPY: y = alpha * x + y
 * ================================================================ */

int pfr_cpu_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    for (int i = 0; i < y->n; i++)
        y->data[i] += alpha * x->data[i];
    return 0;
}
