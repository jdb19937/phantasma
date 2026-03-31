/*
 * pfr_computo_plat.c — dispatch computationis platformae
 * ========================================================
 *
 * Defini PFR_COMPUTO_CPU ut CPU tantum adhibeas.
 * In Linux sine CUDA: CPU semper.
 * In Darwin: Metal temptatur; si non adest, CPU fallback.
 * In Linux cum CUDA: pfr_computo_cuda.cu directe compilatur pro
 * pfr_computo.o — hic plica non adhibetur.
 */

#include <stdlib.h>
#include <string.h>

#include "pfr_computo.h"

/* CPU fallback: semper includitur */
#include "pfr_computo_cpu.c"

#if defined(__APPLE__) && !defined(PHANTASMA_X11) && !defined(PFR_COMPUTO_CPU)
#include "pfr_computo_metal.m"
#else

/* ================================================================
 * Linux (sine CUDA) vel PFR_COMPUTO_CPU: CPU tantum
 * ================================================================ */

int pfr_computo_initia(void) { return 1; /* CPU fallback */ }
void pfr_computo_fini(void) {}

pfr_matrix_t *pfr_matrix_crea(int m, int n)
{
    pfr_matrix_t *a = (pfr_matrix_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->m    = m;
    a->n    = n;
    a->data = (float *)calloc((size_t)m * n, sizeof(float));
    a->gpu  = NULL;
    if (!a->data) { free(a); return NULL; }
    return a;
}

void pfr_matrix_destrue(pfr_matrix_t *a)
{
    if (!a) return;
    free(a->data);
    free(a);
}

pfr_vector_t *pfr_vector_crea(int n)
{
    pfr_vector_t *v = (pfr_vector_t *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->n    = n;
    v->data = (float *)calloc((size_t)n, sizeof(float));
    v->gpu  = NULL;
    if (!v->data) { free(v); return NULL; }
    return v;
}

void pfr_vector_destrue(pfr_vector_t *v)
{
    if (!v) return;
    free(v->data);
    free(v);
}

int pfr_in_gpu_mitte(pfr_matrix_t *a)   { (void)a; return 0; }
int pfr_ex_gpu_cape(pfr_matrix_t *a)    { (void)a; return 0; }
int pfr_in_gpu_mitte_v(pfr_vector_t *v) { (void)v; return 0; }
int pfr_ex_gpu_cape_v(pfr_vector_t *v)  { (void)v; return 0; }

int pfr_matmat(pfr_matrix_t *c, const pfr_matrix_t *a, const pfr_matrix_t *b)
{ return pfr_cpu_matmat(c, a, b); }

int pfr_matvec(pfr_vector_t *y, const pfr_matrix_t *a, const pfr_vector_t *x)
{ return pfr_cpu_matvec(y, a, x); }

int pfr_dotum(float *res, const pfr_vector_t *x, const pfr_vector_t *y)
{ return pfr_cpu_dotum(res, x, y); }

int pfr_scalare(pfr_vector_t *x, float alpha)
{ return pfr_cpu_scalare(x, alpha); }

int pfr_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{ return pfr_cpu_axpy(y, alpha, x); }

/* gpu_* semper fallunt: nullus codicis GPU nexus */
int pfr_gpu_matmat(pfr_matrix_t *c, const pfr_matrix_t *a, const pfr_matrix_t *b)
{ (void)c; (void)a; (void)b; return -1; }

int pfr_gpu_matvec(pfr_vector_t *y, const pfr_matrix_t *a, const pfr_vector_t *x)
{ (void)y; (void)a; (void)x; return -1; }

int pfr_gpu_dotum(float *res, const pfr_vector_t *x, const pfr_vector_t *y)
{ (void)res; (void)x; (void)y; return -1; }

int pfr_gpu_scalare(pfr_vector_t *x, float alpha)
{ (void)x; (void)alpha; return -1; }

int pfr_gpu_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{ (void)y; (void)alpha; (void)x; return -1; }

#endif
