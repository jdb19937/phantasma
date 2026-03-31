/*
 * computo.h — computatio GPU/CPU
 * ================================
 *
 * Operationes BLAS (GEMM, GEMV, DOT, SCAL, AXPY) in GPU
 * (Metal vel CUDA) vel CPU fallback.
 * Duo genera: _f (float simplex) et _d (double duplex).
 *
 * Usus:
 *   pfr_computo_initia() — 0 = GPU, 1 = CPU fallback
 *   pfr_matrix_crea_f / pfr_vector_crea_f — alloca float in CPU
 *   pfr_matrix_crea_d / pfr_vector_crea_d — alloca double in CPU
 *   pfr_in_gpu_mitte_f / pfr_ex_gpu_cape_f — translatio CPU<->GPU (float)
 *   pfr_in_gpu_mitte_d / pfr_ex_gpu_cape_d — translatio CPU<->GPU (double)
 *   pfr_matmat_f / pfr_matvec_f / ... — operationes float
 *   pfr_matmat_d / pfr_matvec_d / ... — operationes double
 */

#ifndef PFR_COMPUTO_H
#define PFR_COMPUTO_H

/* ================================================================
 * typi
 * ================================================================ */

typedef struct pfr_matrix_f {
    int     m;      /* numeri linearum */
    int     n;      /* numeri columnarum */
    float  *data;   /* data in CPU (ordo linearum) */
    void   *gpu;    /* manubrium GPU (opacum; NULL si non in GPU) */
} pfr_matrix_f_t;

typedef struct pfr_vector_f {
    int     n;      /* longitudo */
    float  *data;   /* data in CPU */
    void   *gpu;    /* manubrium GPU (opacum) */
} pfr_vector_f_t;

typedef struct pfr_matrix_d {
    int     m;      /* numeri linearum */
    int     n;      /* numeri columnarum */
    double *data;   /* data in CPU (ordo linearum) */
    void   *gpu;    /* manubrium GPU (opacum; NULL si non in GPU) */
} pfr_matrix_d_t;

typedef struct pfr_vector_d {
    int     n;      /* longitudo */
    double *data;   /* data in CPU */
    void   *gpu;    /* manubrium GPU (opacum) */
} pfr_vector_d_t;

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
 * crea / destrue (float)
 * ================================================================ */

pfr_matrix_f_t *pfr_matrix_crea_f(int m, int n);
void            pfr_matrix_destrue_f(pfr_matrix_f_t *a);
pfr_vector_f_t *pfr_vector_crea_f(int n);
void            pfr_vector_destrue_f(pfr_vector_f_t *v);

/* ================================================================
 * crea / destrue (double)
 * ================================================================ */

pfr_matrix_d_t *pfr_matrix_crea_d(int m, int n);
void            pfr_matrix_destrue_d(pfr_matrix_d_t *a);
pfr_vector_d_t *pfr_vector_crea_d(int n);
void            pfr_vector_destrue_d(pfr_vector_d_t *v);

/* ================================================================
 * translatio CPU <-> GPU (float)
 * ================================================================ */

int pfr_in_gpu_mitte_f(pfr_matrix_f_t *a);     /* CPU -> GPU */
int pfr_ex_gpu_cape_f(pfr_matrix_f_t *a);      /* GPU -> CPU */
int pfr_in_gpu_mitte_vf(pfr_vector_f_t *v);    /* CPU -> GPU */
int pfr_ex_gpu_cape_vf(pfr_vector_f_t *v);     /* GPU -> CPU */

/* ================================================================
 * translatio CPU <-> GPU (double)
 * ================================================================ */

int pfr_in_gpu_mitte_d(pfr_matrix_d_t *a);     /* CPU -> GPU */
int pfr_ex_gpu_cape_d(pfr_matrix_d_t *a);      /* GPU -> CPU */
int pfr_in_gpu_mitte_vd(pfr_vector_d_t *v);    /* CPU -> GPU */
int pfr_ex_gpu_cape_vd(pfr_vector_d_t *v);     /* GPU -> CPU */

/* ================================================================
 * operationes (float)
 * ================================================================ */

/* C = A * B  (GEMM) */
int pfr_matmat_f(pfr_matrix_f_t *c,
                 const pfr_matrix_f_t *a, const pfr_matrix_f_t *b);

/* y = A * x  (GEMV) */
int pfr_matvec_f(pfr_vector_f_t *y,
                 const pfr_matrix_f_t *a, const pfr_vector_f_t *x);

/* *res = x . y  (DOT) */
int pfr_dotum_f(float *res,
                const pfr_vector_f_t *x, const pfr_vector_f_t *y);

/* x = alpha * x  (SCAL) */
int pfr_scalare_f(pfr_vector_f_t *x, float alpha);

/* y = alpha * x + y  (AXPY) */
int pfr_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x);

/* y = A^T * x  (GEMV transpose: y[j] = sum_i A[i,j] * x[i]) */
int pfr_matvec_trans_f(pfr_vector_f_t *y,
                       const pfr_matrix_f_t *a, const pfr_vector_f_t *x);

/* A += alpha * x * y^T  (GER: A[i,j] += alpha * x[i] * y[j]) */
int pfr_ger_f(pfr_matrix_f_t *a, float alpha,
              const pfr_vector_f_t *x, const pfr_vector_f_t *y);

/* ================================================================
 * operationes (double)
 * ================================================================ */

/* C = A * B  (GEMM) */
int pfr_matmat_d(pfr_matrix_d_t *c,
                 const pfr_matrix_d_t *a, const pfr_matrix_d_t *b);

/* y = A * x  (GEMV) */
int pfr_matvec_d(pfr_vector_d_t *y,
                 const pfr_matrix_d_t *a, const pfr_vector_d_t *x);

/* *res = x . y  (DOT) */
int pfr_dotum_d(double *res,
                const pfr_vector_d_t *x, const pfr_vector_d_t *y);

/* x = alpha * x  (SCAL) */
int pfr_scalare_d(pfr_vector_d_t *x, double alpha);

/* y = alpha * x + y  (AXPY) */
int pfr_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x);

/* ================================================================
 * pfr_gpu_* float — fallit (-1) si GPU non adest vel data non in GPU
 * ================================================================ */

int pfr_gpu_matmat_f(pfr_matrix_f_t *c,
                     const pfr_matrix_f_t *a, const pfr_matrix_f_t *b);
int pfr_gpu_matvec_f(pfr_vector_f_t *y,
                     const pfr_matrix_f_t *a, const pfr_vector_f_t *x);
int pfr_gpu_dotum_f(float *res,
                    const pfr_vector_f_t *x, const pfr_vector_f_t *y);
int pfr_gpu_scalare_f(pfr_vector_f_t *x, float alpha);
int pfr_gpu_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x);
int pfr_gpu_matvec_trans_f(pfr_vector_f_t *y,
                           const pfr_matrix_f_t *a, const pfr_vector_f_t *x);
int pfr_gpu_ger_f(pfr_matrix_f_t *a, float alpha,
                  const pfr_vector_f_t *x, const pfr_vector_f_t *y);

/* ================================================================
 * pfr_gpu_* double — fallit (-1) si GPU non adest vel data non in GPU
 * ================================================================ */

int pfr_gpu_matmat_d(pfr_matrix_d_t *c,
                     const pfr_matrix_d_t *a, const pfr_matrix_d_t *b);
int pfr_gpu_matvec_d(pfr_vector_d_t *y,
                     const pfr_matrix_d_t *a, const pfr_vector_d_t *x);
int pfr_gpu_dotum_d(double *res,
                    const pfr_vector_d_t *x, const pfr_vector_d_t *y);
int pfr_gpu_scalare_d(pfr_vector_d_t *x, double alpha);
int pfr_gpu_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x);

/* ================================================================
 * pfr_cpu_* float — semper in CPU, data non in GPU requiritur
 * ================================================================ */

int pfr_cpu_matmat_f(pfr_matrix_f_t *c,
                     const pfr_matrix_f_t *a, const pfr_matrix_f_t *b);
int pfr_cpu_matvec_f(pfr_vector_f_t *y,
                     const pfr_matrix_f_t *a, const pfr_vector_f_t *x);
int pfr_cpu_dotum_f(float *res,
                    const pfr_vector_f_t *x, const pfr_vector_f_t *y);
int pfr_cpu_scalare_f(pfr_vector_f_t *x, float alpha);
int pfr_cpu_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x);
int pfr_cpu_matvec_trans_f(pfr_vector_f_t *y,
                           const pfr_matrix_f_t *a, const pfr_vector_f_t *x);
int pfr_cpu_ger_f(pfr_matrix_f_t *a, float alpha,
                  const pfr_vector_f_t *x, const pfr_vector_f_t *y);

/* ================================================================
 * pfr_cpu_* double — semper in CPU, data non in GPU requiritur
 * ================================================================ */

int pfr_cpu_matmat_d(pfr_matrix_d_t *c,
                     const pfr_matrix_d_t *a, const pfr_matrix_d_t *b);
int pfr_cpu_matvec_d(pfr_vector_d_t *y,
                     const pfr_matrix_d_t *a, const pfr_vector_d_t *x);
int pfr_cpu_dotum_d(double *res,
                    const pfr_vector_d_t *x, const pfr_vector_d_t *y);
int pfr_cpu_scalare_d(pfr_vector_d_t *x, double alpha);
int pfr_cpu_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x);

#endif /* PFR_COMPUTO_H */
