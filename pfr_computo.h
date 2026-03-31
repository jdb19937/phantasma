/*
 * pfr_computo.h — PFR: computatio GPU/CPU
 * =========================================
 *
 * Operationes BLAS (GEMM, GEMV, DOT, SCAL, AXPY) in GPU
 * (Metal vel CUDA) vel CPU fallback.
 *
 * Usus:
 *   pfr_computo_initia() — 0 = GPU, 1 = CPU fallback
 *   pfr_matrix_crea / pfr_vector_crea — alloca in CPU
 *   pfr_in_pfr_gpu_mitte / pfr_ex_pfr_gpu_cape — translatio CPU<->GPU
 *   pfr_matmat / pfr_matvec / pfr_dotum / pfr_scalare / pfr_axpy
 */

#ifndef PFR_COMPUTO_H
#define PFR_COMPUTO_H

/* ================================================================
 * typi
 * ================================================================ */

typedef struct pfr_matrix {
    int     m;      /* numeri linearum */
    int     n;      /* numeri columnarum */
    float  *data;   /* data in CPU (ordo linearum) */
    void   *gpu;    /* manubrium GPU (opacum; NULL si non in GPU) */
} pfr_matrix_t;

typedef struct pfr_vector {
    int     n;      /* longitudo */
    float  *data;   /* data in CPU */
    void   *gpu;    /* manubrium GPU (opacum) */
} pfr_vector_t;

/* ================================================================
 * initium / finis
 * ================================================================ */

/*
 * pfr_computo_initia — initia statum computationis.
 * reddit 0 si GPU adsit, 1 si CPU fallback.
 */
int  pfr_computo_initia(void);
void pfr_computo_fini(void);

/* ================================================================
 * crea / destrue
 * ================================================================ */

pfr_matrix_t *pfr_matrix_crea(int m, int n);
void          pfr_matrix_destrue(pfr_matrix_t *a);
pfr_vector_t *pfr_vector_crea(int n);
void          pfr_vector_destrue(pfr_vector_t *v);

/* ================================================================
 * translatio CPU <-> GPU
 * ================================================================ */

int pfr_in_gpu_mitte(pfr_matrix_t *a);      /* CPU -> GPU */
int pfr_ex_gpu_cape(pfr_matrix_t *a);       /* GPU -> CPU */
int pfr_in_gpu_mitte_v(pfr_vector_t *v);    /* CPU -> GPU */
int pfr_ex_gpu_cape_v(pfr_vector_t *v);     /* GPU -> CPU */

/* ================================================================
 * operationes
 * ================================================================ */

/* C = A * B  (GEMM) */
int pfr_matmat(pfr_matrix_t *c,
               const pfr_matrix_t *a, const pfr_matrix_t *b);

/* y = A * x  (GEMV) */
int pfr_matvec(pfr_vector_t *y,
               const pfr_matrix_t *a, const pfr_vector_t *x);

/* *res = x . y  (DOT) */
int pfr_dotum(float *res,
              const pfr_vector_t *x, const pfr_vector_t *y);

/* x = alpha * x  (SCAL) */
int pfr_scalare(pfr_vector_t *x, float alpha);

/* y = alpha * x + y  (AXPY) */
int pfr_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x);

/* ================================================================
 * pfr_gpu_* — fallit (-1) si GPU non adest vel data non in GPU
 * ================================================================ */

int pfr_gpu_matmat(pfr_matrix_t *c,
               const pfr_matrix_t *a, const pfr_matrix_t *b);
int pfr_gpu_matvec(pfr_vector_t *y,
               const pfr_matrix_t *a, const pfr_vector_t *x);
int pfr_gpu_dotum(float *res,
              const pfr_vector_t *x, const pfr_vector_t *y);
int pfr_gpu_scalare(pfr_vector_t *x, float alpha);
int pfr_gpu_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x);

/* ================================================================
 * pfr_cpu_* — semper in CPU, data non in GPU requiritur
 * ================================================================ */

int pfr_cpu_matmat(pfr_matrix_t *c,
               const pfr_matrix_t *a, const pfr_matrix_t *b);
int pfr_cpu_matvec(pfr_vector_t *y,
               const pfr_matrix_t *a, const pfr_vector_t *x);
int pfr_cpu_dotum(float *res,
              const pfr_vector_t *x, const pfr_vector_t *y);
int pfr_cpu_scalare(pfr_vector_t *x, float alpha);
int pfr_cpu_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x);

#endif /* PFR_COMPUTO_H */
