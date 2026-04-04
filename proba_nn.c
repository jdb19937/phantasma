/*
 * proba_nn.c — benchmark primitiva neuralium retium: GPU vs CPU
 *
 * Comparat pfr_gpu_*, pfr_cpu_*, et pfr_* (genericum) pro variis
 * dimensionibus. Demonstrat GPU accelerationem pro dimensionibus magnis.
 */

#include "computo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double tempus_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void imple_temere(float *p, int n, unsigned int *semen)
{
    for (int i = 0; i < n; i++) {
        *semen = *semen * 1664525u + 1013904223u;
        p[i]   = ((float)(int)*semen) / 2147483648.0f;
    }
}

/* ================================================================
 * benchmark unius operationis
 * ================================================================ */

typedef struct {
    const char *nomen;
    int dim;
    int iterationes;
    double cpu_us;    /* microsecunda per iterationem */
    double gpu_us;
    double gen_us;    /* pfr_* genericum */
} resultatum_t;

static resultatum_t proba_rmsnorm(int dim, int iter)
{
    resultatum_t r = { "rmsnorm", dim, iter, 0, 0, 0 };
    unsigned int semen = 42;
    pfr_vector_f_t *o  = pfr_vector_crea_f(dim);
    pfr_vector_f_t *x  = pfr_vector_crea_f(dim);
    pfr_vector_f_t *w  = pfr_vector_crea_f(dim);
    imple_temere(x->data, dim, &semen);
    imple_temere(w->data, dim, &semen);

    /* CPU */
    double t0 = tempus_sec();
    for (int i = 0; i < iter; i++)
        pfr_cpu_rmsnorm_f(o, x, w, 1e-5f);
    r.cpu_us = (tempus_sec() - t0) / iter * 1e6;

    /* GPU */
    pfr_in_gpu_mitte_vf(x);
    pfr_in_gpu_mitte_vf(w);
    pfr_in_gpu_mitte_vf(o);
    if (o->gpu) {
        /* calefactio */
        pfr_gpu_rmsnorm_f(o, x, w, 1e-5f);
        t0 = tempus_sec();
        for (int i = 0; i < iter; i++)
            pfr_gpu_rmsnorm_f(o, x, w, 1e-5f);
        r.gpu_us = (tempus_sec() - t0) / iter * 1e6;
    }

    /* genericum (debet GPU eligere si adest) */
    t0 = tempus_sec();
    for (int i = 0; i < iter; i++)
        pfr_rmsnorm_f(o, x, w, 1e-5f);
    r.gen_us = (tempus_sec() - t0) / iter * 1e6;

    pfr_vector_destrue_f(o);
    pfr_vector_destrue_f(x);
    pfr_vector_destrue_f(w);
    return r;
}

static resultatum_t proba_matvec(int dim, int iter)
{
    resultatum_t r = { "matvec", dim, iter, 0, 0, 0 };
    unsigned int semen = 42;
    pfr_matrix_f_t *A  = pfr_matrix_crea_f(dim, dim);
    pfr_vector_f_t *x  = pfr_vector_crea_f(dim);
    pfr_vector_f_t *y  = pfr_vector_crea_f(dim);
    imple_temere(A->data, dim * dim, &semen);
    imple_temere(x->data, dim, &semen);

    double t0 = tempus_sec();
    for (int i = 0; i < iter; i++)
        pfr_cpu_matvec_f(y, A, x);
    r.cpu_us = (tempus_sec() - t0) / iter * 1e6;

    pfr_in_gpu_mitte_f(A);
    pfr_in_gpu_mitte_vf(x);
    pfr_in_gpu_mitte_vf(y);
    if (y->gpu) {
        pfr_gpu_matvec_f(y, A, x);
        t0 = tempus_sec();
        for (int i = 0; i < iter; i++)
            pfr_gpu_matvec_f(y, A, x);
        r.gpu_us = (tempus_sec() - t0) / iter * 1e6;
    }

    t0 = tempus_sec();
    for (int i = 0; i < iter; i++)
        pfr_matvec_f(y, A, x);
    r.gen_us = (tempus_sec() - t0) / iter * 1e6;

    pfr_matrix_destrue_f(A);
    pfr_vector_destrue_f(x);
    pfr_vector_destrue_f(y);
    return r;
}

static resultatum_t proba_softmax(int dim, int iter)
{
    resultatum_t r = { "softmax", dim, iter, 0, 0, 0 };
    unsigned int semen = 42;
    pfr_vector_f_t *x  = pfr_vector_crea_f(dim);
    imple_temere(x->data, dim, &semen);

    double t0 = tempus_sec();
    for (int i = 0; i < iter; i++) {
        imple_temere(x->data, dim, &semen);
        pfr_cpu_softmax_f(x);
    }
    r.cpu_us = (tempus_sec() - t0) / iter * 1e6;

    pfr_in_gpu_mitte_vf(x);
    if (x->gpu) {
        imple_temere(x->data, dim, &semen);
        pfr_in_gpu_mitte_vf(x);
        pfr_gpu_softmax_f(x);
        t0 = tempus_sec();
        for (int i = 0; i < iter; i++) {
            pfr_gpu_softmax_f(x);
        }
        r.gpu_us = (tempus_sec() - t0) / iter * 1e6;
    }

    t0 = tempus_sec();
    for (int i = 0; i < iter; i++)
        pfr_softmax_f(x);
    r.gen_us = (tempus_sec() - t0) / iter * 1e6;

    pfr_vector_destrue_f(x);
    return r;
}

static resultatum_t proba_swiglu(int dim, int iter)
{
    resultatum_t r = { "swiglu", dim, iter, 0, 0, 0 };
    unsigned int semen = 42;
    pfr_vector_f_t *o  = pfr_vector_crea_f(dim);
    pfr_vector_f_t *a  = pfr_vector_crea_f(dim);
    pfr_vector_f_t *b  = pfr_vector_crea_f(dim);
    imple_temere(a->data, dim, &semen);
    imple_temere(b->data, dim, &semen);

    double t0 = tempus_sec();
    for (int i = 0; i < iter; i++)
        pfr_cpu_swiglu_f(o, a, b);
    r.cpu_us = (tempus_sec() - t0) / iter * 1e6;

    pfr_in_gpu_mitte_vf(a);
    pfr_in_gpu_mitte_vf(b);
    pfr_in_gpu_mitte_vf(o);
    if (o->gpu) {
        pfr_gpu_swiglu_f(o, a, b);
        t0 = tempus_sec();
        for (int i = 0; i < iter; i++)
            pfr_gpu_swiglu_f(o, a, b);
        r.gpu_us = (tempus_sec() - t0) / iter * 1e6;
    }

    t0 = tempus_sec();
    for (int i = 0; i < iter; i++)
        pfr_swiglu_f(o, a, b);
    r.gen_us = (tempus_sec() - t0) / iter * 1e6;

    pfr_vector_destrue_f(o);
    pfr_vector_destrue_f(a);
    pfr_vector_destrue_f(b);
    return r;
}

/* ================================================================ */

static void imprime(resultatum_t r)
{
    printf(
        "  %-8s dim=%-5d  cpu=%8.1f us  gpu=%8.1f us  gen=%8.1f us",
        r.nomen, r.dim, r.cpu_us, r.gpu_us, r.gen_us
    );
    if (r.gpu_us > 0 && r.cpu_us > 0) {
        double ratio = r.cpu_us / r.gpu_us;
        printf(
            "  %s %.1fx",
            ratio > 1.0 ? "GPU velocior" : "CPU velocior",
            ratio > 1.0 ? ratio : 1.0 / ratio
        );
    }
    printf("\n");
}

int main(void)
{
    int gpu = pfr_computo_initia();
    printf(
        "=== proba_nn: GPU %s ===\n\n",
        gpu == 0 ? "adest" : "non adest (CPU solum)"
    );

    int dims[] = { 64, 128, 256, 512, 1024, 2048, 4096 };
    int n_dims = 7;

    for (int d = 0; d < n_dims; d++) {
        int dim  = dims[d];
        int iter = dim <= 256 ? 1000 : (dim <= 1024 ? 200 : 50);
        printf("dim = %d (%d iterationes):\n", dim, iter);
        imprime(proba_rmsnorm(dim, iter));
        imprime(proba_matvec(dim, iter));
        imprime(proba_softmax(dim, iter));
        imprime(proba_swiglu(dim, iter));
        printf("\n");
    }

    pfr_computo_fini();
    return 0;
}
