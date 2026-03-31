/*
 * computo_plat.c — dispatch computationis platformae
 * ========================================================
 *
 * Defini PFR_COMPUTO_CPU ut CPU tantum adhibeas.
 * In Linux sine CUDA: CPU semper.
 * In Darwin: Metal temptatur; si non adest, CPU fallback.
 * In Linux cum CUDA: computo_cuda.cu directe compilatur pro
 * computo.o — hic plica non adhibetur.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "computo.h"

/* CPU fallback: semper includitur */
#include "computo_cpu.c"

/* attentio: semper CPU (raw pointers, compositum non BLAS) */

int pfr_attentio_f(float *o, const float *q,
                   const float *cache_k, const float *cache_v,
                   float *att,
                   int d, int n_capita, int n_capita_kv,
                   int positio, int longitudo_max)
{ return pfr_cpu_attentio_f(o, q, cache_k, cache_v, att, d, n_capita,
                             n_capita_kv, positio, longitudo_max); }

#if defined(__APPLE__) && !defined(PHANTASMA_X11) && !defined(PFR_COMPUTO_CPU)
#include "computo_metal.m"
#else

/* ================================================================
 * Linux (sine CUDA) vel PFR_COMPUTO_CPU: CPU tantum
 * ================================================================ */

int pfr_computo_initia(void) { return 1; /* CPU fallback */ }
void pfr_computo_fini(void) {}

pfr_matrix_f_t *pfr_matrix_crea_f(int m, int n)
{
    pfr_matrix_f_t *a = (pfr_matrix_f_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->m    = m;
    a->n    = n;
    a->data = (float *)calloc((size_t)m * n, sizeof(float));
    a->gpu  = NULL;
    if (!a->data) { free(a); return NULL; }
    return a;
}

void pfr_matrix_destrue_f(pfr_matrix_f_t *a)
{
    if (!a) return;
    free(a->data);
    free(a);
}

pfr_vector_f_t *pfr_vector_crea_f(int n)
{
    pfr_vector_f_t *v = (pfr_vector_f_t *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->n    = n;
    v->data = (float *)calloc((size_t)n, sizeof(float));
    v->gpu  = NULL;
    if (!v->data) { free(v); return NULL; }
    return v;
}

void pfr_vector_destrue_f(pfr_vector_f_t *v)
{
    if (!v) return;
    free(v->data);
    free(v);
}

pfr_matrix_d_t *pfr_matrix_crea_d(int m, int n)
{
    pfr_matrix_d_t *a = (pfr_matrix_d_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->m    = m;
    a->n    = n;
    a->data = (double *)calloc((size_t)m * n, sizeof(double));
    a->gpu  = NULL;
    if (!a->data) { free(a); return NULL; }
    return a;
}

void pfr_matrix_destrue_d(pfr_matrix_d_t *a)
{
    if (!a) return;
    free(a->data);
    free(a);
}

pfr_vector_d_t *pfr_vector_crea_d(int n)
{
    pfr_vector_d_t *v = (pfr_vector_d_t *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->n    = n;
    v->data = (double *)calloc((size_t)n, sizeof(double));
    v->gpu  = NULL;
    if (!v->data) { free(v); return NULL; }
    return v;
}

void pfr_vector_destrue_d(pfr_vector_d_t *v)
{
    if (!v) return;
    free(v->data);
    free(v);
}

int pfr_in_gpu_mitte_f(pfr_matrix_f_t *a)    { (void)a; return 0; }
int pfr_ex_gpu_cape_f(pfr_matrix_f_t *a)     { (void)a; return 0; }
int pfr_in_gpu_mitte_vf(pfr_vector_f_t *v)   { (void)v; return 0; }
int pfr_ex_gpu_cape_vf(pfr_vector_f_t *v)    { (void)v; return 0; }

int pfr_in_gpu_mitte_d(pfr_matrix_d_t *a)    { (void)a; return 0; }
int pfr_ex_gpu_cape_d(pfr_matrix_d_t *a)     { (void)a; return 0; }
int pfr_in_gpu_mitte_vd(pfr_vector_d_t *v)   { (void)v; return 0; }
int pfr_ex_gpu_cape_vd(pfr_vector_d_t *v)    { (void)v; return 0; }

int pfr_matmat_f(pfr_matrix_f_t *c,
                 const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{ return pfr_cpu_matmat_f(c, a, b); }

int pfr_matvec_f(pfr_vector_f_t *y,
                 const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{ return pfr_cpu_matvec_f(y, a, x); }

int pfr_dotum_f(float *res,
                const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{ return pfr_cpu_dotum_f(res, x, y); }

int pfr_scalare_f(pfr_vector_f_t *x, float alpha)
{ return pfr_cpu_scalare_f(x, alpha); }

int pfr_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{ return pfr_cpu_axpy_f(y, alpha, x); }

int pfr_matvec_trans_f(pfr_vector_f_t *y,
                       const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{ return pfr_cpu_matvec_trans_f(y, a, x); }

int pfr_ger_f(pfr_matrix_f_t *a, float alpha,
              const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{ return pfr_cpu_ger_f(a, alpha, x, y); }

int pfr_matmat_d(pfr_matrix_d_t *c,
                 const pfr_matrix_d_t *a, const pfr_matrix_d_t *b)
{ return pfr_cpu_matmat_d(c, a, b); }

int pfr_matvec_d(pfr_vector_d_t *y,
                 const pfr_matrix_d_t *a, const pfr_vector_d_t *x)
{ return pfr_cpu_matvec_d(y, a, x); }

int pfr_dotum_d(double *res,
                const pfr_vector_d_t *x, const pfr_vector_d_t *y)
{ return pfr_cpu_dotum_d(res, x, y); }

int pfr_scalare_d(pfr_vector_d_t *x, double alpha)
{ return pfr_cpu_scalare_d(x, alpha); }

int pfr_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x)
{ return pfr_cpu_axpy_d(y, alpha, x); }

/* gpu_* semper fallunt: nullus codicis GPU nexus */
int pfr_gpu_matmat_f(pfr_matrix_f_t *c,
                     const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{ (void)c; (void)a; (void)b; return -1; }

int pfr_gpu_matvec_f(pfr_vector_f_t *y,
                     const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{ (void)y; (void)a; (void)x; return -1; }

int pfr_gpu_dotum_f(float *res,
                    const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{ (void)res; (void)x; (void)y; return -1; }

int pfr_gpu_scalare_f(pfr_vector_f_t *x, float alpha)
{ (void)x; (void)alpha; return -1; }

int pfr_gpu_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{ (void)y; (void)alpha; (void)x; return -1; }

int pfr_gpu_matvec_trans_f(pfr_vector_f_t *y,
                           const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{ (void)y; (void)a; (void)x; return -1; }

int pfr_gpu_ger_f(pfr_matrix_f_t *a, float alpha,
                  const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{ (void)a; (void)alpha; (void)x; (void)y; return -1; }

int pfr_gpu_matmat_d(pfr_matrix_d_t *c,
                     const pfr_matrix_d_t *a, const pfr_matrix_d_t *b)
{ (void)c; (void)a; (void)b; return -1; }

int pfr_gpu_matvec_d(pfr_vector_d_t *y,
                     const pfr_matrix_d_t *a, const pfr_vector_d_t *x)
{ (void)y; (void)a; (void)x; return -1; }

int pfr_gpu_dotum_d(double *res,
                    const pfr_vector_d_t *x, const pfr_vector_d_t *y)
{ (void)res; (void)x; (void)y; return -1; }

int pfr_gpu_scalare_d(pfr_vector_d_t *x, double alpha)
{ (void)x; (void)alpha; return -1; }

int pfr_gpu_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x)
{ (void)y; (void)alpha; (void)x; return -1; }

/* primitiva neuralium retium: pfr_* = CPU, pfr_gpu_* = fallit */

int pfr_rmsnorm_f(pfr_vector_f_t *o, const pfr_vector_f_t *x,
                  const pfr_vector_f_t *w, float eps)
{ return pfr_cpu_rmsnorm_f(o, x, w, eps); }

int pfr_swiglu_f(pfr_vector_f_t *o, const pfr_vector_f_t *a,
                 const pfr_vector_f_t *b)
{ return pfr_cpu_swiglu_f(o, a, b); }

int pfr_softmax_f(pfr_vector_f_t *x)
{ return pfr_cpu_softmax_f(x); }

int pfr_rope_f(pfr_vector_f_t *v, int positio)
{ return pfr_cpu_rope_f(v, positio); }

int pfr_gpu_rmsnorm_f(pfr_vector_f_t *o, const pfr_vector_f_t *x,
                      const pfr_vector_f_t *w, float eps)
{ (void)o; (void)x; (void)w; (void)eps; return -1; }

int pfr_gpu_swiglu_f(pfr_vector_f_t *o, const pfr_vector_f_t *a,
                     const pfr_vector_f_t *b)
{ (void)o; (void)a; (void)b; return -1; }

int pfr_gpu_softmax_f(pfr_vector_f_t *x)
{ (void)x; return -1; }

int pfr_gpu_rope_f(pfr_vector_f_t *v, int positio)
{ (void)v; (void)positio; return -1; }

#endif
