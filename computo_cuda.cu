/*
 * computo_cuda.cu — computatio CUDA (Linux)
 * ================================================
 *
 * Compilatur per nvcc directe pro computo.o.
 * Si nulla machina CUDA invenitur in initio, usar_gpu = 0
 * et omnes operationes ad pfr_cpu_* cadunt.
 *
 * Notae precisionis:
 *   _f (float): GPU et CPU.
 *   _d (double): GPU et CPU — CUDA precisionem duplicem sustinet.
 */

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#include "computo.h"

/* CPU fallback — nexu C ut nomina non deturpentur */
#include <math.h>
extern "C" {
#include "computo_cpu.c"
}

/* ================================================================
 * status globalis
 * ================================================================ */

static struct {
    int usar_gpu;
    int id_machinae;
} cc;

/* ================================================================
 * nuclei CUDA (float)
 * ================================================================ */

__global__ static void kern_matmat_f(const float *A, const float *B,
                                      float *C, int m, int n, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    float s = 0.0f;
    for (int l = 0; l < k; l++)
        s += A[i*k + l] * B[l*n + j];
    C[i*n + j] = s;
}

__global__ static void kern_matvec_f(const float *A, const float *x,
                                      float *y, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float s = 0.0f;
    for (int j = 0; j < n; j++)
        s += A[i*n + j] * x[j];
    y[i] = s;
}

__global__ static void kern_dotum_f(const float *x, const float *y,
                                     float *partes, int n)
{
    extern __shared__ float suma[];
    int tid = (int)threadIdx.x;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    suma[tid] = (gid < n) ? x[gid] * y[gid] : 0.0f;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) suma[tid] += suma[tid + s];
        __syncthreads();
    }
    if (tid == 0) partes[blockIdx.x] = suma[0];
}

__global__ static void kern_scalare_f(float *x, float alpha, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[i] *= alpha;
}

__global__ static void kern_axpy_f(float *y, float alpha,
                                    const float *x, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] += alpha * x[i];
}

/* ================================================================
 * nuclei CUDA: primitiva neuralium retium (float)
 * ================================================================ */

__global__ static void kern_rmsnorm_f(const float *x, const float *w,
                                       float *o, int n, float eps)
{
    extern __shared__ float scrip[];
    int tid = (int)threadIdx.x;
    int tgs = (int)blockDim.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += tgs) ss += x[i] * x[i];
    scrip[tid] = ss;
    __syncthreads();
    for (int s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) scrip[tid] += scrip[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(scrip[0] / (float)n + eps);
    __syncthreads();
    for (int i = tid; i < n; i += tgs)
        o[i] = w[i] * x[i] * inv_rms;
}

__global__ static void kern_swiglu_f(const float *a, const float *b,
                                      float *o, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float v = a[i];
    float sig = 1.0f / (1.0f + expf(-v));
    o[i] = v * sig * b[i];
}

__global__ static void kern_softmax_f(float *x, int n)
{
    extern __shared__ float scrip[];
    int tid = (int)threadIdx.x;
    int tgs = (int)blockDim.x;
    float mx = -1e30f;
    for (int i = tid; i < n; i += tgs)
        if (x[i] > mx) mx = x[i];
    scrip[tid] = mx;
    __syncthreads();
    for (int s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s && scrip[tid + s] > scrip[tid])
            scrip[tid] = scrip[tid + s];
        __syncthreads();
    }
    mx = scrip[0];
    __syncthreads();
    float ss = 0.0f;
    for (int i = tid; i < n; i += tgs) {
        x[i] = expf(x[i] - mx);
        ss += x[i];
    }
    scrip[tid] = ss;
    __syncthreads();
    for (int s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) scrip[tid] += scrip[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / scrip[0];
    __syncthreads();
    for (int i = tid; i < n; i += tgs)
        x[i] *= inv;
}

__global__ static void kern_rope_f(float *v, int n, int pos)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;
    float freq = 1.0f / powf(10000.0f, (float)i / (float)n);
    float val = (float)pos * freq;
    float fcr = cosf(val), fci = sinf(val);
    float v0 = v[i], v1 = v[i + 1];
    v[i]     = v0 * fcr - v1 * fci;
    v[i + 1] = v0 * fci + v1 * fcr;
}

/* ================================================================
 * nuclei CUDA (double)
 * ================================================================ */

__global__ static void kern_matmat_d(const double *A, const double *B,
                                      double *C, int m, int n, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    double s = 0.0;
    for (int l = 0; l < k; l++)
        s += A[i*k + l] * B[l*n + j];
    C[i*n + j] = s;
}

__global__ static void kern_matvec_d(const double *A, const double *x,
                                      double *y, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double s = 0.0;
    for (int j = 0; j < n; j++)
        s += A[i*n + j] * x[j];
    y[i] = s;
}

__global__ static void kern_dotum_d(const double *x, const double *y,
                                     double *partes, int n)
{
    extern __shared__ double suma_d[];
    int tid = (int)threadIdx.x;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    suma_d[tid] = (gid < n) ? x[gid] * y[gid] : 0.0;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) suma_d[tid] += suma_d[tid + s];
        __syncthreads();
    }
    if (tid == 0) partes[blockIdx.x] = suma_d[0];
}

__global__ static void kern_scalare_d(double *x, double alpha, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[i] *= alpha;
}

__global__ static void kern_axpy_d(double *y, double alpha,
                                    const double *x, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] += alpha * x[i];
}

/* ================================================================
 * initium / finis
 * ================================================================ */

extern "C" int pfr_computo_initia(void)
{
    cc.usar_gpu = 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 1;
    cc.id_machinae = 0;
    if (cudaSetDevice(cc.id_machinae) != cudaSuccess) return 1;
    cc.usar_gpu = 1;
    return 0;
}

extern "C" void pfr_computo_fini(void)
{
    if (cc.usar_gpu)
        cudaDeviceReset();
    cc.usar_gpu = 0;
}

/* ================================================================
 * crea / destrue (float)
 * ================================================================ */

extern "C" pfr_matrix_f_t *pfr_matrix_crea_f(int m, int n)
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

extern "C" void pfr_matrix_destrue_f(pfr_matrix_f_t *a)
{
    if (!a) return;
    if (a->gpu) { cudaFree(a->gpu); a->gpu = NULL; }
    free(a->data);
    free(a);
}

extern "C" pfr_vector_f_t *pfr_vector_crea_f(int n)
{
    pfr_vector_f_t *v = (pfr_vector_f_t *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->n    = n;
    v->data = (float *)calloc((size_t)n, sizeof(float));
    v->gpu  = NULL;
    if (!v->data) { free(v); return NULL; }
    return v;
}

extern "C" void pfr_vector_destrue_f(pfr_vector_f_t *v)
{
    if (!v) return;
    if (v->gpu) { cudaFree(v->gpu); v->gpu = NULL; }
    free(v->data);
    free(v);
}

/* ================================================================
 * crea / destrue (double)
 * ================================================================ */

extern "C" pfr_matrix_d_t *pfr_matrix_crea_d(int m, int n)
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

extern "C" void pfr_matrix_destrue_d(pfr_matrix_d_t *a)
{
    if (!a) return;
    if (a->gpu) { cudaFree(a->gpu); a->gpu = NULL; }
    free(a->data);
    free(a);
}

extern "C" pfr_vector_d_t *pfr_vector_crea_d(int n)
{
    pfr_vector_d_t *v = (pfr_vector_d_t *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->n    = n;
    v->data = (double *)calloc((size_t)n, sizeof(double));
    v->gpu  = NULL;
    if (!v->data) { free(v); return NULL; }
    return v;
}

extern "C" void pfr_vector_destrue_d(pfr_vector_d_t *v)
{
    if (!v) return;
    if (v->gpu) { cudaFree(v->gpu); v->gpu = NULL; }
    free(v->data);
    free(v);
}

/* ================================================================
 * translatio CPU <-> GPU (float)
 * ================================================================ */

extern "C" int pfr_in_gpu_mitte_f(pfr_matrix_f_t *a)
{
    if (!a || !cc.usar_gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(float);
    if (!a->gpu) {
        if (cudaMalloc(&a->gpu, bytes) != cudaSuccess) return -1;
    }
    if (cudaMemcpy(a->gpu, a->data, bytes, cudaMemcpyHostToDevice)
            != cudaSuccess) return -1;
    return 0;
}

extern "C" int pfr_ex_gpu_cape_f(pfr_matrix_f_t *a)
{
    if (!a || !a->gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(float);
    if (cudaMemcpy(a->data, a->gpu, bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) return -1;
    return 0;
}

extern "C" int pfr_in_gpu_mitte_vf(pfr_vector_f_t *v)
{
    if (!v || !cc.usar_gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(float);
    if (!v->gpu) {
        if (cudaMalloc(&v->gpu, bytes) != cudaSuccess) return -1;
    }
    if (cudaMemcpy(v->gpu, v->data, bytes, cudaMemcpyHostToDevice)
            != cudaSuccess) return -1;
    return 0;
}

extern "C" int pfr_ex_gpu_cape_vf(pfr_vector_f_t *v)
{
    if (!v || !v->gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(float);
    if (cudaMemcpy(v->data, v->gpu, bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) return -1;
    return 0;
}

/* ================================================================
 * translatio CPU <-> GPU (double)
 * ================================================================ */

extern "C" int pfr_in_gpu_mitte_d(pfr_matrix_d_t *a)
{
    if (!a || !cc.usar_gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(double);
    if (!a->gpu) {
        if (cudaMalloc(&a->gpu, bytes) != cudaSuccess) return -1;
    }
    if (cudaMemcpy(a->gpu, a->data, bytes, cudaMemcpyHostToDevice)
            != cudaSuccess) return -1;
    return 0;
}

extern "C" int pfr_ex_gpu_cape_d(pfr_matrix_d_t *a)
{
    if (!a || !a->gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(double);
    if (cudaMemcpy(a->data, a->gpu, bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) return -1;
    return 0;
}

extern "C" int pfr_in_gpu_mitte_vd(pfr_vector_d_t *v)
{
    if (!v || !cc.usar_gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(double);
    if (!v->gpu) {
        if (cudaMalloc(&v->gpu, bytes) != cudaSuccess) return -1;
    }
    if (cudaMemcpy(v->gpu, v->data, bytes, cudaMemcpyHostToDevice)
            != cudaSuccess) return -1;
    return 0;
}

extern "C" int pfr_ex_gpu_cape_vd(pfr_vector_d_t *v)
{
    if (!v || !v->gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(double);
    if (cudaMemcpy(v->data, v->gpu, bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) return -1;
    return 0;
}

/* ================================================================
 * internae CUDA float
 * ================================================================ */

static int cuda_matmat_f(pfr_matrix_f_t *c,
                          const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{
    int m = a->m, n = b->n, k = a->n;
    dim3 tgs(16, 16);
    dim3 grd((n+15)/16, (m+15)/16);
    kern_matmat_f<<<grd,tgs>>>((float*)a->gpu,(float*)b->gpu,(float*)c->gpu,m,n,k);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_matvec_f(pfr_vector_f_t *y,
                          const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    int m = a->m, n = a->n, tgs = 256;
    kern_matvec_f<<<(m+tgs-1)/tgs,tgs>>>((float*)a->gpu,(float*)x->gpu,
                                           (float*)y->gpu,m,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_dotum_f(float *res,
                         const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    int n = x->n, tgs = 256, n_grupe = (n+tgs-1)/tgs;
    float *pg = NULL;
    if (cudaMalloc(&pg, n_grupe * sizeof(float)) != cudaSuccess) return -1;
    kern_dotum_f<<<n_grupe,tgs,(size_t)tgs*sizeof(float)>>>(
        (float*)x->gpu,(float*)y->gpu,pg,n);
    cudaDeviceSynchronize();
    float *pc = (float *)malloc(n_grupe * sizeof(float));
    cudaMemcpy(pc, pg, n_grupe*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pg);
    float s = 0.0f;
    for (int i = 0; i < n_grupe; i++) s += pc[i];
    free(pc);
    *res = s;
    return 0;
}

static int cuda_scalare_f(pfr_vector_f_t *x, float alpha)
{
    int n = x->n, tgs = 256;
    kern_scalare_f<<<(n+tgs-1)/tgs,tgs>>>((float*)x->gpu,alpha,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{
    int n = y->n, tgs = 256;
    kern_axpy_f<<<(n+tgs-1)/tgs,tgs>>>((float*)y->gpu,alpha,(float*)x->gpu,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

/* ================================================================
 * internae CUDA: primitiva neuralium retium (float)
 * ================================================================ */

static int cuda_rmsnorm_f(void *o_gpu, void *x_gpu, void *w_gpu,
                           int n, float eps)
{
    int tgs = 256;
    if (n < tgs) tgs = n;
    int t2 = 1;
    while (t2 < tgs) t2 <<= 1;
    tgs = t2;
    kern_rmsnorm_f<<<1, tgs, tgs * sizeof(float)>>>(
        (float*)x_gpu, (float*)w_gpu, (float*)o_gpu, n, eps);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_swiglu_f(void *o_gpu, void *a_gpu, void *b_gpu, int n)
{
    int tgs = 256;
    kern_swiglu_f<<<(n+tgs-1)/tgs, tgs>>>(
        (float*)a_gpu, (float*)b_gpu, (float*)o_gpu, n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_softmax_f(void *x_gpu, int n)
{
    int tgs = 256;
    if (n < tgs) tgs = n;
    int t2 = 1;
    while (t2 < tgs) t2 <<= 1;
    tgs = t2;
    kern_softmax_f<<<1, tgs, tgs * sizeof(float)>>>((float*)x_gpu, n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_rope_f(void *v_gpu, int n, int pos)
{
    int tgs = 256;
    int paria = (n / 2 + tgs - 1) / tgs;
    kern_rope_f<<<paria, tgs>>>((float*)v_gpu, n, pos);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

/* ================================================================
 * internae CUDA double
 * ================================================================ */

static int cuda_matmat_d(pfr_matrix_d_t *c,
                          const pfr_matrix_d_t *a, const pfr_matrix_d_t *b)
{
    int m = a->m, n = b->n, k = a->n;
    dim3 tgs(16, 16);
    dim3 grd((n+15)/16, (m+15)/16);
    kern_matmat_d<<<grd,tgs>>>((double*)a->gpu,(double*)b->gpu,(double*)c->gpu,m,n,k);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_matvec_d(pfr_vector_d_t *y,
                          const pfr_matrix_d_t *a, const pfr_vector_d_t *x)
{
    int m = a->m, n = a->n, tgs = 256;
    kern_matvec_d<<<(m+tgs-1)/tgs,tgs>>>((double*)a->gpu,(double*)x->gpu,
                                           (double*)y->gpu,m,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_dotum_d(double *res,
                         const pfr_vector_d_t *x, const pfr_vector_d_t *y)
{
    int n = x->n, tgs = 256, n_grupe = (n+tgs-1)/tgs;
    double *pg = NULL;
    if (cudaMalloc(&pg, n_grupe * sizeof(double)) != cudaSuccess) return -1;
    kern_dotum_d<<<n_grupe,tgs,(size_t)tgs*sizeof(double)>>>(
        (double*)x->gpu,(double*)y->gpu,pg,n);
    cudaDeviceSynchronize();
    double *pc = (double *)malloc(n_grupe * sizeof(double));
    cudaMemcpy(pc, pg, n_grupe*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(pg);
    double s = 0.0;
    for (int i = 0; i < n_grupe; i++) s += pc[i];
    free(pc);
    *res = s;
    return 0;
}

static int cuda_scalare_d(pfr_vector_d_t *x, double alpha)
{
    int n = x->n, tgs = 256;
    kern_scalare_d<<<(n+tgs-1)/tgs,tgs>>>((double*)x->gpu,alpha,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x)
{
    int n = y->n, tgs = 256;
    kern_axpy_d<<<(n+tgs-1)/tgs,tgs>>>((double*)y->gpu,alpha,(double*)x->gpu,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

/* ================================================================
 * pfr_*_f — conatur GPU; cadit ad CPU si non adest
 * ================================================================ */

extern "C" int pfr_matmat_f(pfr_matrix_f_t *c,
                              const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!cc.usar_gpu || !a->gpu || !b->gpu || !c->gpu)
        return pfr_cpu_matmat_f(c, a, b);
    return cuda_matmat_f(c, a, b);
}

extern "C" int pfr_matvec_f(pfr_vector_f_t *y,
                              const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!cc.usar_gpu || !a->gpu || !x->gpu || !y->gpu)
        return pfr_cpu_matvec_f(y, a, x);
    return cuda_matvec_f(y, a, x);
}

extern "C" int pfr_dotum_f(float *res,
                             const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!cc.usar_gpu || !x->gpu || !y->gpu)
        return pfr_cpu_dotum_f(res, x, y);
    return cuda_dotum_f(res, x, y);
}

extern "C" int pfr_scalare_f(pfr_vector_f_t *x, float alpha)
{
    if (!x) return -1;
    if (!cc.usar_gpu || !x->gpu) return pfr_cpu_scalare_f(x, alpha);
    return cuda_scalare_f(x, alpha);
}

extern "C" int pfr_axpy_f(pfr_vector_f_t *y, float alpha,
                           const pfr_vector_f_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!cc.usar_gpu || !y->gpu || !x->gpu) return pfr_cpu_axpy_f(y, alpha, x);
    return cuda_axpy_f(y, alpha, x);
}

/* matvec_trans et ger: CPU solum (nuclei CUDA nondum implementati) */

extern "C" int pfr_matvec_trans_f(pfr_vector_f_t *y,
                                    const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{ return pfr_cpu_matvec_trans_f(y, a, x); }

extern "C" int pfr_ger_f(pfr_matrix_f_t *a, float alpha,
                           const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{ return pfr_cpu_ger_f(a, alpha, x, y); }

/* primitiva neuralium retium: conatur GPU, cadit ad CPU */

extern "C" int pfr_rmsnorm_f(pfr_vector_f_t *o, const pfr_vector_f_t *x,
                               const pfr_vector_f_t *w, float eps)
{
    if (!o || !x || !w) return -1;
    if (cc.usar_gpu && o->gpu && x->gpu && w->gpu)
        return cuda_rmsnorm_f(o->gpu, x->gpu, w->gpu, x->n, eps);
    return pfr_cpu_rmsnorm_f(o, x, w, eps);
}

extern "C" int pfr_swiglu_f(pfr_vector_f_t *o, const pfr_vector_f_t *a,
                              const pfr_vector_f_t *b)
{
    if (!o || !a || !b) return -1;
    if (cc.usar_gpu && o->gpu && a->gpu && b->gpu)
        return cuda_swiglu_f(o->gpu, a->gpu, b->gpu, a->n);
    return pfr_cpu_swiglu_f(o, a, b);
}

extern "C" int pfr_softmax_f(pfr_vector_f_t *x)
{
    if (!x) return -1;
    if (cc.usar_gpu && x->gpu)
        return cuda_softmax_f(x->gpu, x->n);
    return pfr_cpu_softmax_f(x);
}

extern "C" int pfr_rope_f(pfr_vector_f_t *v, int positio)
{
    if (!v) return -1;
    if (cc.usar_gpu && v->gpu)
        return cuda_rope_f(v->gpu, v->n, positio);
    return pfr_cpu_rope_f(v, positio);
}

extern "C" int pfr_attentio_f(float *o, const float *q,
                                const float *cache_k, const float *cache_v,
                                float *att,
                                int d, int n_capita, int n_capita_kv,
                                int positio, int longitudo_max)
{ return pfr_cpu_attentio_f(o, q, cache_k, cache_v, att, d, n_capita,
                              n_capita_kv, positio, longitudo_max); }

/* ================================================================
 * pfr_*_d — conatur GPU; cadit ad CPU si non adest
 * ================================================================ */

extern "C" int pfr_matmat_d(pfr_matrix_d_t *c,
                              const pfr_matrix_d_t *a, const pfr_matrix_d_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!cc.usar_gpu || !a->gpu || !b->gpu || !c->gpu)
        return pfr_cpu_matmat_d(c, a, b);
    return cuda_matmat_d(c, a, b);
}

extern "C" int pfr_matvec_d(pfr_vector_d_t *y,
                              const pfr_matrix_d_t *a, const pfr_vector_d_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!cc.usar_gpu || !a->gpu || !x->gpu || !y->gpu)
        return pfr_cpu_matvec_d(y, a, x);
    return cuda_matvec_d(y, a, x);
}

extern "C" int pfr_dotum_d(double *res,
                             const pfr_vector_d_t *x, const pfr_vector_d_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!cc.usar_gpu || !x->gpu || !y->gpu)
        return pfr_cpu_dotum_d(res, x, y);
    return cuda_dotum_d(res, x, y);
}

extern "C" int pfr_scalare_d(pfr_vector_d_t *x, double alpha)
{
    if (!x) return -1;
    if (!cc.usar_gpu || !x->gpu) return pfr_cpu_scalare_d(x, alpha);
    return cuda_scalare_d(x, alpha);
}

extern "C" int pfr_axpy_d(pfr_vector_d_t *y, double alpha,
                           const pfr_vector_d_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!cc.usar_gpu || !y->gpu || !x->gpu) return pfr_cpu_axpy_d(y, alpha, x);
    return cuda_axpy_d(y, alpha, x);
}

/* ================================================================
 * gpu_*_f — fallit si GPU non adest vel data non in GPU
 * ================================================================ */

extern "C" int pfr_gpu_matmat_f(pfr_matrix_f_t *c,
                                  const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!cc.usar_gpu || !a->gpu || !b->gpu || !c->gpu) return -1;
    return cuda_matmat_f(c, a, b);
}

extern "C" int pfr_gpu_matvec_f(pfr_vector_f_t *y,
                                  const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!cc.usar_gpu || !a->gpu || !x->gpu || !y->gpu) return -1;
    return cuda_matvec_f(y, a, x);
}

extern "C" int pfr_gpu_dotum_f(float *res,
                                 const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!cc.usar_gpu || !x->gpu || !y->gpu) return -1;
    return cuda_dotum_f(res, x, y);
}

extern "C" int pfr_gpu_scalare_f(pfr_vector_f_t *x, float alpha)
{
    if (!x || !cc.usar_gpu || !x->gpu) return -1;
    return cuda_scalare_f(x, alpha);
}

extern "C" int pfr_gpu_axpy_f(pfr_vector_f_t *y, float alpha,
                                const pfr_vector_f_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!cc.usar_gpu || !y->gpu || !x->gpu) return -1;
    return cuda_axpy_f(y, alpha, x);
}

extern "C" int pfr_gpu_matvec_trans_f(pfr_vector_f_t *y,
                                       const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{ (void)y; (void)a; (void)x; return -1; }

extern "C" int pfr_gpu_ger_f(pfr_matrix_f_t *a, float alpha,
                               const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{ (void)a; (void)alpha; (void)x; (void)y; return -1; }

/* primitiva neuralium retium: GPU solum */

extern "C" int pfr_gpu_rmsnorm_f(pfr_vector_f_t *o, const pfr_vector_f_t *x,
                                   const pfr_vector_f_t *w, float eps)
{
    if (!o || !x || !w) return -1;
    if (!cc.usar_gpu || !o->gpu || !x->gpu || !w->gpu) return -1;
    return cuda_rmsnorm_f(o->gpu, x->gpu, w->gpu, x->n, eps);
}

extern "C" int pfr_gpu_swiglu_f(pfr_vector_f_t *o, const pfr_vector_f_t *a,
                                  const pfr_vector_f_t *b)
{
    if (!o || !a || !b) return -1;
    if (!cc.usar_gpu || !o->gpu || !a->gpu || !b->gpu) return -1;
    return cuda_swiglu_f(o->gpu, a->gpu, b->gpu, a->n);
}

extern "C" int pfr_gpu_softmax_f(pfr_vector_f_t *x)
{
    if (!x) return -1;
    if (!cc.usar_gpu || !x->gpu) return -1;
    return cuda_softmax_f(x->gpu, x->n);
}

extern "C" int pfr_gpu_rope_f(pfr_vector_f_t *v, int positio)
{
    if (!v) return -1;
    if (!cc.usar_gpu || !v->gpu) return -1;
    return cuda_rope_f(v->gpu, v->n, positio);
}

/* ================================================================
 * gpu_*_d — fallit si GPU non adest vel data non in GPU
 * ================================================================ */

extern "C" int pfr_gpu_matmat_d(pfr_matrix_d_t *c,
                                  const pfr_matrix_d_t *a, const pfr_matrix_d_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!cc.usar_gpu || !a->gpu || !b->gpu || !c->gpu) return -1;
    return cuda_matmat_d(c, a, b);
}

extern "C" int pfr_gpu_matvec_d(pfr_vector_d_t *y,
                                  const pfr_matrix_d_t *a, const pfr_vector_d_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!cc.usar_gpu || !a->gpu || !x->gpu || !y->gpu) return -1;
    return cuda_matvec_d(y, a, x);
}

extern "C" int pfr_gpu_dotum_d(double *res,
                                 const pfr_vector_d_t *x, const pfr_vector_d_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!cc.usar_gpu || !x->gpu || !y->gpu) return -1;
    return cuda_dotum_d(res, x, y);
}

extern "C" int pfr_gpu_scalare_d(pfr_vector_d_t *x, double alpha)
{
    if (!x || !cc.usar_gpu || !x->gpu) return -1;
    return cuda_scalare_d(x, alpha);
}

extern "C" int pfr_gpu_axpy_d(pfr_vector_d_t *y, double alpha,
                                const pfr_vector_d_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!cc.usar_gpu || !y->gpu || !x->gpu) return -1;
    return cuda_axpy_d(y, alpha, x);
}
