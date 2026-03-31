/*
 * computo_cuda.cu — computatio CUDA (Linux)
 * ================================================
 *
 * Compilatur per nvcc directe pro computo.o.
 * Si nulla machina CUDA invenitur in initio, usar_gpu = 0
 * et omnes operationes ad pfr_cpu_* cadunt.
 */

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#include "computo.h"

/* CPU fallback — nexu C ut nomina non deturpentur */
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
 * nuclei CUDA
 * ================================================================ */

__global__ static void kern_matmat(const float *A, const float *B,
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

__global__ static void kern_matvec(const float *A, const float *x,
                                    float *y, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float s = 0.0f;
    for (int j = 0; j < n; j++)
        s += A[i*n + j] * x[j];
    y[i] = s;
}

/* reducere per threadgroup -> partes[bid] */
__global__ static void kern_dotum(const float *x, const float *y,
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

__global__ static void kern_scalare(float *x, float alpha, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[i] *= alpha;
}

__global__ static void kern_axpy(float *y, float alpha,
                                  const float *x, int n)
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
 * crea / destrue
 * ================================================================ */

extern "C" pfr_matrix_t *pfr_matrix_crea(int m, int n)
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

extern "C" void pfr_matrix_destrue(pfr_matrix_t *a)
{
    if (!a) return;
    if (a->gpu) { cudaFree(a->gpu); a->gpu = NULL; }
    free(a->data);
    free(a);
}

extern "C" pfr_vector_t *pfr_vector_crea(int n)
{
    pfr_vector_t *v = (pfr_vector_t *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->n    = n;
    v->data = (float *)calloc((size_t)n, sizeof(float));
    v->gpu  = NULL;
    if (!v->data) { free(v); return NULL; }
    return v;
}

extern "C" void pfr_vector_destrue(pfr_vector_t *v)
{
    if (!v) return;
    if (v->gpu) { cudaFree(v->gpu); v->gpu = NULL; }
    free(v->data);
    free(v);
}

/* ================================================================
 * translatio CPU <-> GPU
 * ================================================================ */

extern "C" int pfr_in_gpu_mitte(pfr_matrix_t *a)
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

extern "C" int pfr_ex_gpu_cape(pfr_matrix_t *a)
{
    if (!a || !a->gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(float);
    if (cudaMemcpy(a->data, a->gpu, bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) return -1;
    return 0;
}

extern "C" int pfr_in_gpu_mitte_v(pfr_vector_t *v)
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

extern "C" int pfr_ex_gpu_cape_v(pfr_vector_t *v)
{
    if (!v || !v->gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(float);
    if (cudaMemcpy(v->data, v->gpu, bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) return -1;
    return 0;
}

/* ================================================================
 * internae CUDA: assumunt cc.usar_gpu et omnes gpu* non-NULL
 * ================================================================ */

static int cuda_matmat(pfr_matrix_t *c,
                        const pfr_matrix_t *a, const pfr_matrix_t *b)
{
    int m = a->m, n = b->n, k = a->n;
    dim3 tgs(16, 16);
    dim3 grd((n+15)/16, (m+15)/16);
    kern_matmat<<<grd,tgs>>>((float*)a->gpu,(float*)b->gpu,(float*)c->gpu,m,n,k);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_matvec(pfr_vector_t *y,
                        const pfr_matrix_t *a, const pfr_vector_t *x)
{
    int m = a->m, n = a->n, tgs = 256;
    kern_matvec<<<(m+tgs-1)/tgs,tgs>>>((float*)a->gpu,(float*)x->gpu,
                                         (float*)y->gpu,m,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_dotum(float *res,
                       const pfr_vector_t *x, const pfr_vector_t *y)
{
    int n = x->n, tgs = 256, n_grupe = (n+tgs-1)/tgs;
    float *pg = NULL;
    if (cudaMalloc(&pg, n_grupe * sizeof(float)) != cudaSuccess) return -1;
    kern_dotum<<<n_grupe,tgs,(size_t)tgs*sizeof(float)>>>(
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

static int cuda_scalare(pfr_vector_t *x, float alpha)
{
    int n = x->n, tgs = 256;
    kern_scalare<<<(n+tgs-1)/tgs,tgs>>>((float*)x->gpu,alpha,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

static int cuda_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{
    int n = y->n, tgs = 256;
    kern_axpy<<<(n+tgs-1)/tgs,tgs>>>((float*)y->gpu,alpha,(float*)x->gpu,n);
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

/* ================================================================
 * pfr_* — conatur GPU; cadit ad CPU si non adest
 * ================================================================ */

extern "C" int pfr_matmat(pfr_matrix_t *c,
                           const pfr_matrix_t *a, const pfr_matrix_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!cc.usar_gpu || !a->gpu || !b->gpu || !c->gpu)
        return pfr_cpu_matmat(c, a, b);
    return cuda_matmat(c, a, b);
}

extern "C" int pfr_matvec(pfr_vector_t *y,
                           const pfr_matrix_t *a, const pfr_vector_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!cc.usar_gpu || !a->gpu || !x->gpu || !y->gpu)
        return pfr_cpu_matvec(y, a, x);
    return cuda_matvec(y, a, x);
}

extern "C" int pfr_dotum(float *res,
                          const pfr_vector_t *x, const pfr_vector_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!cc.usar_gpu || !x->gpu || !y->gpu)
        return pfr_cpu_dotum(res, x, y);
    return cuda_dotum(res, x, y);
}

extern "C" int pfr_scalare(pfr_vector_t *x, float alpha)
{
    if (!x) return -1;
    if (!cc.usar_gpu || !x->gpu) return pfr_cpu_scalare(x, alpha);
    return cuda_scalare(x, alpha);
}

extern "C" int pfr_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!cc.usar_gpu || !y->gpu || !x->gpu) return pfr_cpu_axpy(y, alpha, x);
    return cuda_axpy(y, alpha, x);
}

/* ================================================================
 * gpu_* — fallit si GPU non adest vel data non in GPU
 * ================================================================ */

extern "C" int pfr_gpu_matmat(pfr_matrix_t *c,
                           const pfr_matrix_t *a, const pfr_matrix_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!cc.usar_gpu || !a->gpu || !b->gpu || !c->gpu) return -1;
    return cuda_matmat(c, a, b);
}

extern "C" int pfr_gpu_matvec(pfr_vector_t *y,
                           const pfr_matrix_t *a, const pfr_vector_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!cc.usar_gpu || !a->gpu || !x->gpu || !y->gpu) return -1;
    return cuda_matvec(y, a, x);
}

extern "C" int pfr_gpu_dotum(float *res,
                          const pfr_vector_t *x, const pfr_vector_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!cc.usar_gpu || !x->gpu || !y->gpu) return -1;
    return cuda_dotum(res, x, y);
}

extern "C" int pfr_gpu_scalare(pfr_vector_t *x, float alpha)
{
    if (!x || !cc.usar_gpu || !x->gpu) return -1;
    return cuda_scalare(x, alpha);
}

extern "C" int pfr_gpu_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!cc.usar_gpu || !y->gpu || !x->gpu) return -1;
    return cuda_axpy(y, alpha, x);
}
