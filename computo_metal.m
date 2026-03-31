/*
 * computo_metal.m — computatio Metal (Darwin)
 * ==================================================
 *
 * Includ. ab computo_plat.c (ut Objective-C compilatur).
 * Parens iam inclusit: computo.h, <stdlib.h>, <string.h>,
 * computo_cpu.c (pfr_cpu_* functiones staticae).
 *
 * Tempore cursus: si MTLCreateSystemDefaultDevice() reddit nil
 * vel compilatio MSL fallit, usar_gpu = 0 et omnes operationes
 * ad pfr_cpu_* cadunt.
 *
 * Notae precisionis:
 *   _f (float): in GPU vel CPU.
 *   _d (double): semper in CPU — Metal precisionem duplicem
 *   in machinis GPGPU non sustinet.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

/* ================================================================
 * fons MSL (Metal Shading Language)
 * ================================================================ */

static const char *fons_msl =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "/* GEMM: C[i,j] = sum_l A[i,l] * B[l,j] */\n"
    "kernel void kern_matmat(\n"
    "    device const float* A [[buffer(0)]],\n"
    "    device const float* B [[buffer(1)]],\n"
    "    device       float* C [[buffer(2)]],\n"
    "    constant int&       m [[buffer(3)]],\n"
    "    constant int&       n [[buffer(4)]],\n"
    "    constant int&       k [[buffer(5)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    int i = (int)gid.y, j = (int)gid.x;\n"
    "    if (i >= m || j >= n) return;\n"
    "    float s = 0.0;\n"
    "    for (int l = 0; l < k; l++)\n"
    "        s += A[i * k + l] * B[l * n + j];\n"
    "    C[i * n + j] = s;\n"
    "}\n"
    "\n"
    "/* GEMV: y[i] = sum_j A[i,j] * x[j] */\n"
    "kernel void kern_matvec(\n"
    "    device const float* A [[buffer(0)]],\n"
    "    device const float* x [[buffer(1)]],\n"
    "    device       float* y [[buffer(2)]],\n"
    "    constant int&       m [[buffer(3)]],\n"
    "    constant int&       n [[buffer(4)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    int i = (int)gid;\n"
    "    if (i >= m) return;\n"
    "    float s = 0.0;\n"
    "    for (int j = 0; j < n; j++)\n"
    "        s += A[i * n + j] * x[j];\n"
    "    y[i] = s;\n"
    "}\n"
    "\n"
    "/* DOT: par[bid] = sum of x[gid]*y[gid] in threadgroup */\n"
    "kernel void kern_dotum(\n"
    "    device const float* x   [[buffer(0)]],\n"
    "    device const float* y   [[buffer(1)]],\n"
    "    device       float* par [[buffer(2)]],\n"
    "    constant int&       n   [[buffer(3)]],\n"
    "    threadgroup float* suma [[threadgroup(0)]],\n"
    "    uint tid [[thread_index_in_threadgroup]],\n"
    "    uint tgs [[threads_per_threadgroup]],\n"
    "    uint gid [[thread_position_in_grid]],\n"
    "    uint bid [[threadgroup_position_in_grid]])\n"
    "{\n"
    "    suma[tid] = ((int)gid < n) ? x[gid] * y[gid] : 0.0;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = tgs >> 1; s > 0; s >>= 1) {\n"
    "        if (tid < s) suma[tid] += suma[tid + s];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (tid == 0) par[bid] = suma[0];\n"
    "}\n"
    "\n"
    "/* SCAL: x[i] *= a */\n"
    "kernel void kern_scalare(\n"
    "    device     float* x [[buffer(0)]],\n"
    "    constant float&   a [[buffer(1)]],\n"
    "    constant int&     n [[buffer(2)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if ((int)gid < n) x[gid] *= a;\n"
    "}\n"
    "\n"
    "/* AXPY: y[i] += alpha * x[i] */\n"
    "kernel void kern_axpy(\n"
    "    device       float* y     [[buffer(0)]],\n"
    "    constant   float&   alpha [[buffer(1)]],\n"
    "    device const float* x     [[buffer(2)]],\n"
    "    constant   int&     n     [[buffer(3)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if ((int)gid < n) y[gid] += alpha * x[gid];\n"
    "}\n"
    "\n"
    "/* GEMV^T: y[j] = sum_i A[i*n+j] * x[i] */\n"
    "kernel void kern_matvec_trans(\n"
    "    device const float* A [[buffer(0)]],\n"
    "    device const float* x [[buffer(1)]],\n"
    "    device       float* y [[buffer(2)]],\n"
    "    constant int&       m [[buffer(3)]],\n"
    "    constant int&       n [[buffer(4)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    int j = (int)gid;\n"
    "    if (j >= n) return;\n"
    "    float s = 0.0;\n"
    "    for (int i = 0; i < m; i++)\n"
    "        s += A[i * n + j] * x[i];\n"
    "    y[j] = s;\n"
    "}\n"
    "\n"
    "/* GER: A[i*n+j] += alpha * x[i] * y[j] */\n"
    "kernel void kern_ger(\n"
    "    device       float* A     [[buffer(0)]],\n"
    "    constant   float&   alpha [[buffer(1)]],\n"
    "    device const float* x     [[buffer(2)]],\n"
    "    device const float* y     [[buffer(3)]],\n"
    "    constant int&       m     [[buffer(4)]],\n"
    "    constant int&       n     [[buffer(5)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    int ij = (int)gid;\n"
    "    int i = ij / n, j = ij % n;\n"
    "    if (i >= m) return;\n"
    "    A[ij] += alpha * x[i] * y[j];\n"
    "}\n"
    "\n"
    "/* RMSNorm: o[j] = w[j] * x[j] * rsqrt(mean(x^2) + eps) */\n"
    "kernel void kern_rmsnorm(\n"
    "    device const float* x [[buffer(0)]],\n"
    "    device const float* w [[buffer(1)]],\n"
    "    device float* o [[buffer(2)]],\n"
    "    constant int& n [[buffer(3)]],\n"
    "    constant float& eps [[buffer(4)]],\n"
    "    threadgroup float* scrip [[threadgroup(0)]],\n"
    "    uint tid [[thread_index_in_threadgroup]],\n"
    "    uint tgs [[threads_per_threadgroup]])\n"
    "{\n"
    "    float ss = 0.0;\n"
    "    for (int i = (int)tid; i < n; i += (int)tgs)\n"
    "        ss += x[i] * x[i];\n"
    "    scrip[tid] = ss;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = tgs >> 1; s > 0; s >>= 1) {\n"
    "        if (tid < s) scrip[tid] += scrip[tid + s];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    float inv_rms = rsqrt(scrip[0] / (float)n + eps);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (int i = (int)tid; i < n; i += (int)tgs)\n"
    "        o[i] = w[i] * x[i] * inv_rms;\n"
    "}\n"
    "\n"
    "/* SwiGLU: o[i] = silu(a[i]) * b[i] */\n"
    "kernel void kern_swiglu(\n"
    "    device const float* a [[buffer(0)]],\n"
    "    device const float* b [[buffer(1)]],\n"
    "    device float* o [[buffer(2)]],\n"
    "    constant int& n [[buffer(3)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if ((int)gid >= n) return;\n"
    "    float v = a[gid];\n"
    "    float sig = 1.0 / (1.0 + exp(-v));\n"
    "    o[gid] = v * sig * b[gid];\n"
    "}\n"
    "\n"
    "/* Softmax in situ */\n"
    "kernel void kern_softmax(\n"
    "    device float* x [[buffer(0)]],\n"
    "    constant int& n [[buffer(1)]],\n"
    "    threadgroup float* scrip [[threadgroup(0)]],\n"
    "    uint tid [[thread_index_in_threadgroup]],\n"
    "    uint tgs [[threads_per_threadgroup]])\n"
    "{\n"
    "    float mx = -1e30;\n"
    "    for (int i = (int)tid; i < n; i += (int)tgs)\n"
    "        mx = max(mx, x[i]);\n"
    "    scrip[tid] = mx;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = tgs >> 1; s > 0; s >>= 1) {\n"
    "        if (tid < s) scrip[tid] = max(scrip[tid], scrip[tid + s]);\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    mx = scrip[0];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    float ss = 0.0;\n"
    "    for (int i = (int)tid; i < n; i += (int)tgs) {\n"
    "        x[i] = exp(x[i] - mx);\n"
    "        ss += x[i];\n"
    "    }\n"
    "    scrip[tid] = ss;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = tgs >> 1; s > 0; s >>= 1) {\n"
    "        if (tid < s) scrip[tid] += scrip[tid + s];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    float inv = 1.0 / scrip[0];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (int i = (int)tid; i < n; i += (int)tgs)\n"
    "        x[i] *= inv;\n"
    "}\n"
    "\n"
    "/* RoPE: rotat paria per positionem */\n"
    "kernel void kern_rope(\n"
    "    device float* v [[buffer(0)]],\n"
    "    constant int& n [[buffer(1)]],\n"
    "    constant int& pos [[buffer(2)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    int i = (int)gid * 2;\n"
    "    if (i >= n) return;\n"
    "    float freq = 1.0 / pow(10000.0, (float)i / (float)n);\n"
    "    float val = (float)pos * freq;\n"
    "    float fcr = cos(val), fci = sin(val);\n"
    "    float v0 = v[i], v1 = v[i + 1];\n"
    "    v[i]     = v0 * fcr - v1 * fci;\n"
    "    v[i + 1] = v0 * fci + v1 * fcr;\n"
    "}\n";

/* ================================================================
 * status globalis
 * ================================================================ */

#define MC_MATMAT       0
#define MC_MATVEC       1
#define MC_DOTUM        2
#define MC_SCALARE      3
#define MC_AXPY         4
#define MC_MATVEC_TRANS 5
#define MC_GER          6
#define MC_RMSNORM      7
#define MC_SWIGLU       8
#define MC_SOFTMAX      9
#define MC_ROPE        10
#define MC_N_PLENA     11

static struct {
    id<MTLDevice>               machina;
    id<MTLCommandQueue>         coda;
    id<MTLComputePipelineState> plena[MC_N_PLENA];
    int                         usar_gpu;
} mc;

/* ================================================================
 * auxiliaria interna
 * ================================================================ */

/*
 * mc_id_buffer — reddit manubrium Metal ex campo gpu (void*).
 * Nullum retentum additur.
 */
static id<MTLBuffer> mc_id_buffer(void *gpu)
{
    return (id<MTLBuffer>)gpu;
}

/*
 * mc_plena_crea — creat statum pipeline pro functione nomine data.
 * reddit nil si error.
 */
static id<MTLComputePipelineState> mc_plena_crea(id<MTLLibrary> bib,
                                                   NSString *nomen)
{
    id<MTLFunction> fn = [bib newFunctionWithName:nomen];
    if (!fn) return nil;
    NSError *err = nil;
    id<MTLComputePipelineState> pl =
        [mc.machina newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    return pl; /* retentum +1 */
}

/* ================================================================
 * initium / finis
 * ================================================================ */

int pfr_computo_initia(void)
{
    mc.usar_gpu = 0;

    mc.machina = MTLCreateSystemDefaultDevice();
    if (!mc.machina) return 1; /* non adest GPU */

    mc.coda = [mc.machina newCommandQueue];
    if (!mc.coda) {
        [mc.machina release];
        mc.machina = nil;
        return 1;
    }

    @autoreleasepool {
        NSError *err = nil;
        id<MTLLibrary> bib = [mc.machina
            newLibraryWithSource:[NSString stringWithUTF8String:fons_msl]
            options:nil
            error:&err];
        if (!bib) {
            [mc.coda release];    mc.coda    = nil;
            [mc.machina release]; mc.machina = nil;
            return 1;
        }

        static const char *nomina[MC_N_PLENA] = {
            "kern_matmat", "kern_matvec", "kern_dotum",
            "kern_scalare", "kern_axpy",
            "kern_matvec_trans", "kern_ger",
            "kern_rmsnorm", "kern_swiglu",
            "kern_softmax", "kern_rope"
        };
        int ok = 1;
        for (int i = 0; i < MC_N_PLENA; i++) {
            mc.plena[i] = mc_plena_crea(bib,
                [NSString stringWithUTF8String:nomina[i]]);
            if (!mc.plena[i]) { ok = 0; break; }
        }
        [bib release];

        if (!ok) {
            for (int i = 0; i < MC_N_PLENA; i++) {
                [mc.plena[i] release];
                mc.plena[i] = nil;
            }
            [mc.coda release];    mc.coda    = nil;
            [mc.machina release]; mc.machina = nil;
            return 1;
        }
    }

    mc.usar_gpu = 1;
    return 0;
}

void pfr_computo_fini(void)
{
    if (!mc.usar_gpu) return;
    for (int i = 0; i < MC_N_PLENA; i++) {
        [mc.plena[i] release];
        mc.plena[i] = nil;
    }
    [mc.coda release];    mc.coda    = nil;
    [mc.machina release]; mc.machina = nil;
    mc.usar_gpu = 0;
}

/* ================================================================
 * crea / destrue (float)
 * ================================================================ */

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
    if (a->gpu) { [mc_id_buffer(a->gpu) release]; a->gpu = NULL; }
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
    if (v->gpu) { [mc_id_buffer(v->gpu) release]; v->gpu = NULL; }
    free(v->data);
    free(v);
}

/* ================================================================
 * crea / destrue (double) — CPU solum, nullum GPU
 * ================================================================ */

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

/* ================================================================
 * translatio CPU <-> GPU (float)
 * ================================================================ */

int pfr_in_gpu_mitte_f(pfr_matrix_f_t *a)
{
    if (!a || !mc.usar_gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(float);
    if (a->gpu) {
        /* buffer iam adest: renova solum */
        memcpy([mc_id_buffer(a->gpu) contents], a->data, bytes);
        return 0;
    }
    id<MTLBuffer> buf = [mc.machina
        newBufferWithBytes:a->data
        length:bytes
        options:MTLResourceStorageModeShared];
    if (!buf) return -1;
    a->gpu = (void *)buf; /* retentum +1 ex new* */
    return 0;
}

int pfr_ex_gpu_cape_f(pfr_matrix_f_t *a)
{
    if (!a || !a->gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(float);
    memcpy(a->data, [mc_id_buffer(a->gpu) contents], bytes);
    return 0;
}

int pfr_in_gpu_mitte_vf(pfr_vector_f_t *v)
{
    if (!v || !mc.usar_gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(float);
    if (v->gpu) {
        memcpy([mc_id_buffer(v->gpu) contents], v->data, bytes);
        return 0;
    }
    id<MTLBuffer> buf = [mc.machina
        newBufferWithBytes:v->data
        length:bytes
        options:MTLResourceStorageModeShared];
    if (!buf) return -1;
    v->gpu = (void *)buf;
    return 0;
}

int pfr_ex_gpu_cape_vf(pfr_vector_f_t *v)
{
    if (!v || !v->gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(float);
    memcpy(v->data, [mc_id_buffer(v->gpu) contents], bytes);
    return 0;
}

/* ================================================================
 * translatio CPU <-> GPU (double) — Metal non sustinet; no-op
 * ================================================================ */

int pfr_in_gpu_mitte_d(pfr_matrix_d_t *a)  { (void)a; return 0; }
int pfr_ex_gpu_cape_d(pfr_matrix_d_t *a)   { (void)a; return 0; }
int pfr_in_gpu_mitte_vd(pfr_vector_d_t *v) { (void)v; return 0; }
int pfr_ex_gpu_cape_vd(pfr_vector_d_t *v)  { (void)v; return 0; }

/* ================================================================
 * auxilium: mandatum synchronum
 * ================================================================ */

static void mc_exsequere(id<MTLCommandBuffer> cb)
{
    [cb commit];
    [cb waitUntilCompleted];
}

/* ================================================================
 * internae Metal: assumunt mc.usar_gpu et omnes gpu* non-NULL
 * ================================================================ */

static int metal_matmat(pfr_matrix_f_t *c,
                         const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{
    int m = a->m, n = b->n, k = a->n;
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_MATMAT]];
        [enc setBuffer:mc_id_buffer(a->gpu) offset:0 atIndex:0];
        [enc setBuffer:mc_id_buffer(b->gpu) offset:0 atIndex:1];
        [enc setBuffer:mc_id_buffer(c->gpu) offset:0 atIndex:2];
        [enc setBytes:&m length:sizeof(int) atIndex:3];
        [enc setBytes:&n length:sizeof(int) atIndex:4];
        [enc setBytes:&k length:sizeof(int) atIndex:5];
        NSUInteger tx = 16, ty = 16;
        MTLSize tgs = MTLSizeMake(tx, ty, 1);
        MTLSize grd = MTLSizeMake(((size_t)n+tx-1)/tx,
                                   ((size_t)m+ty-1)/ty, 1);
        [enc dispatchThreadgroups:grd threadsPerThreadgroup:tgs];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_matvec(pfr_vector_f_t *y,
                         const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    int m = a->m, n = a->n;
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_MATVEC]];
        [enc setBuffer:mc_id_buffer(a->gpu) offset:0 atIndex:0];
        [enc setBuffer:mc_id_buffer(x->gpu) offset:0 atIndex:1];
        [enc setBuffer:mc_id_buffer(y->gpu) offset:0 atIndex:2];
        [enc setBytes:&m length:sizeof(int) atIndex:3];
        [enc setBytes:&n length:sizeof(int) atIndex:4];
        NSUInteger tgs = 256;
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(((size_t)m+tgs-1)/tgs, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_dotum(float *res,
                        const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    int n = x->n;
    NSUInteger tgs_n  = 256;
    NSUInteger n_grupe = ((size_t)n + tgs_n - 1) / tgs_n;
    @autoreleasepool {
        id<MTLBuffer> par = [mc.machina
            newBufferWithLength:n_grupe * sizeof(float)
            options:MTLResourceStorageModeShared];
        if (!par) return -1;
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_DOTUM]];
        [enc setBuffer:mc_id_buffer(x->gpu) offset:0 atIndex:0];
        [enc setBuffer:mc_id_buffer(y->gpu) offset:0 atIndex:1];
        [enc setBuffer:par                  offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(int) atIndex:3];
        [enc setThreadgroupMemoryLength:tgs_n * sizeof(float) atIndex:0];
        MTLSize t = MTLSizeMake(tgs_n, 1, 1);
        MTLSize g = MTLSizeMake(n_grupe, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
        float *partes = (float *)[par contents];
        float s = 0.0f;
        for (NSUInteger i = 0; i < n_grupe; i++) s += partes[i];
        *res = s;
        [par release];
    }
    return 0;
}

static int metal_scalare(pfr_vector_f_t *x, float alpha)
{
    int n = x->n;
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_SCALARE]];
        [enc setBuffer:mc_id_buffer(x->gpu) offset:0 atIndex:0];
        [enc setBytes:&alpha length:sizeof(float) atIndex:1];
        [enc setBytes:&n     length:sizeof(int)   atIndex:2];
        NSUInteger tgs = 256;
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(((size_t)n+tgs-1)/tgs, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_axpy(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{
    int n = y->n;
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_AXPY]];
        [enc setBuffer:mc_id_buffer(y->gpu) offset:0 atIndex:0];
        [enc setBytes:&alpha length:sizeof(float) atIndex:1];
        [enc setBuffer:mc_id_buffer(x->gpu) offset:0 atIndex:2];
        [enc setBytes:&n     length:sizeof(int)   atIndex:3];
        NSUInteger tgs = 256;
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(((size_t)n+tgs-1)/tgs, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_matvec_trans(pfr_vector_f_t *y,
                              const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    int m = a->m, n = a->n;
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_MATVEC_TRANS]];
        [enc setBuffer:mc_id_buffer(a->gpu) offset:0 atIndex:0];
        [enc setBuffer:mc_id_buffer(x->gpu) offset:0 atIndex:1];
        [enc setBuffer:mc_id_buffer(y->gpu) offset:0 atIndex:2];
        [enc setBytes:&m length:sizeof(int) atIndex:3];
        [enc setBytes:&n length:sizeof(int) atIndex:4];
        NSUInteger tgs = 256;
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(((size_t)n+tgs-1)/tgs, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_ger(pfr_matrix_f_t *a, float alpha,
                     const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    int m = a->m, n = a->n;
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_GER]];
        [enc setBuffer:mc_id_buffer(a->gpu) offset:0 atIndex:0];
        [enc setBytes:&alpha length:sizeof(float) atIndex:1];
        [enc setBuffer:mc_id_buffer(x->gpu) offset:0 atIndex:2];
        [enc setBuffer:mc_id_buffer(y->gpu) offset:0 atIndex:3];
        [enc setBytes:&m length:sizeof(int) atIndex:4];
        [enc setBytes:&n length:sizeof(int) atIndex:5];
        NSUInteger tgs = 256;
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(((size_t)m*n+tgs-1)/tgs, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

/* ================================================================
 * internae Metal: primitiva neuralium retium
 * ================================================================ */

static int metal_rmsnorm(void *o_gpu, void *x_gpu, void *w_gpu,
                         int n, float eps)
{
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_RMSNORM]];
        [enc setBuffer:mc_id_buffer(x_gpu) offset:0 atIndex:0];
        [enc setBuffer:mc_id_buffer(w_gpu) offset:0 atIndex:1];
        [enc setBuffer:mc_id_buffer(o_gpu) offset:0 atIndex:2];
        [enc setBytes:&n   length:sizeof(int)   atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        NSUInteger tgs = 256;
        if ((NSUInteger)n < tgs) tgs = (NSUInteger)n;
        NSUInteger t2 = 1;
        while (t2 < tgs) t2 <<= 1;
        tgs = t2;
        [enc setThreadgroupMemoryLength:tgs * sizeof(float) atIndex:0];
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_swiglu(void *o_gpu, void *a_gpu, void *b_gpu, int n)
{
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_SWIGLU]];
        [enc setBuffer:mc_id_buffer(a_gpu) offset:0 atIndex:0];
        [enc setBuffer:mc_id_buffer(b_gpu) offset:0 atIndex:1];
        [enc setBuffer:mc_id_buffer(o_gpu) offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(int) atIndex:3];
        NSUInteger tgs = 256;
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(((size_t)n + tgs - 1) / tgs, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_softmax(void *x_gpu, int n)
{
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_SOFTMAX]];
        [enc setBuffer:mc_id_buffer(x_gpu) offset:0 atIndex:0];
        [enc setBytes:&n length:sizeof(int) atIndex:1];
        NSUInteger tgs = 256;
        if ((NSUInteger)n < tgs) tgs = (NSUInteger)n;
        NSUInteger t2 = 1;
        while (t2 < tgs) t2 <<= 1;
        tgs = t2;
        [enc setThreadgroupMemoryLength:tgs * sizeof(float) atIndex:0];
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

static int metal_rope(void *v_gpu, int n, int pos)
{
    @autoreleasepool {
        id<MTLCommandBuffer>         cb  = [mc.coda commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mc.plena[MC_ROPE]];
        [enc setBuffer:mc_id_buffer(v_gpu) offset:0 atIndex:0];
        [enc setBytes:&n   length:sizeof(int) atIndex:1];
        [enc setBytes:&pos length:sizeof(int) atIndex:2];
        NSUInteger tgs = 256;
        NSUInteger paria = ((size_t)n / 2 + tgs - 1) / tgs;
        MTLSize t = MTLSizeMake(tgs, 1, 1);
        MTLSize g = MTLSizeMake(paria, 1, 1);
        [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
        [enc endEncoding];
        mc_exsequere(cb);
    }
    return 0;
}

/* ================================================================
 * pfr_*_f — conatur GPU; cadit ad CPU si non adest
 * ================================================================ */

int pfr_matmat_f(pfr_matrix_f_t *c,
                 const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !b->gpu || !c->gpu)
        return pfr_cpu_matmat_f(c, a, b);
    return metal_matmat(c, a, b);
}

int pfr_matvec_f(pfr_vector_f_t *y,
                 const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu)
        return pfr_cpu_matvec_f(y, a, x);
    return metal_matvec(y, a, x);
}

int pfr_dotum_f(float *res, const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!mc.usar_gpu || !x->gpu || !y->gpu)
        return pfr_cpu_dotum_f(res, x, y);
    return metal_dotum(res, x, y);
}

int pfr_scalare_f(pfr_vector_f_t *x, float alpha)
{
    if (!x) return -1;
    if (!mc.usar_gpu || !x->gpu) return pfr_cpu_scalare_f(x, alpha);
    return metal_scalare(x, alpha);
}

int pfr_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!mc.usar_gpu || !y->gpu || !x->gpu) return pfr_cpu_axpy_f(y, alpha, x);
    return metal_axpy(y, alpha, x);
}

int pfr_matvec_trans_f(pfr_vector_f_t *y,
                       const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->m != x->n || y->n != a->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu)
        return pfr_cpu_matvec_trans_f(y, a, x);
    return metal_matvec_trans(y, a, x);
}

int pfr_ger_f(pfr_matrix_f_t *a, float alpha,
              const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!a || !x || !y) return -1;
    if (a->m != x->n || a->n != y->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu)
        return pfr_cpu_ger_f(a, alpha, x, y);
    return metal_ger(a, alpha, x, y);
}

/* ================================================================
 * pfr_*_d — semper CPU (Metal precisionem duplicem non sustinet)
 * ================================================================ */

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

/* ================================================================
 * gpu_*_f — fallit si GPU non adest vel data non in GPU
 * ================================================================ */

int pfr_gpu_matmat_f(pfr_matrix_f_t *c,
                     const pfr_matrix_f_t *a, const pfr_matrix_f_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !b->gpu || !c->gpu) return -1;
    return metal_matmat(c, a, b);
}

int pfr_gpu_matvec_f(pfr_vector_f_t *y,
                     const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu) return -1;
    return metal_matvec(y, a, x);
}

int pfr_gpu_dotum_f(float *res,
                    const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!mc.usar_gpu || !x->gpu || !y->gpu) return -1;
    return metal_dotum(res, x, y);
}

int pfr_gpu_scalare_f(pfr_vector_f_t *x, float alpha)
{
    if (!x || !mc.usar_gpu || !x->gpu) return -1;
    return metal_scalare(x, alpha);
}

int pfr_gpu_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!mc.usar_gpu || !y->gpu || !x->gpu) return -1;
    return metal_axpy(y, alpha, x);
}

int pfr_gpu_matvec_trans_f(pfr_vector_f_t *y,
                           const pfr_matrix_f_t *a, const pfr_vector_f_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->m != x->n || y->n != a->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu) return -1;
    return metal_matvec_trans(y, a, x);
}

int pfr_gpu_ger_f(pfr_matrix_f_t *a, float alpha,
                  const pfr_vector_f_t *x, const pfr_vector_f_t *y)
{
    if (!a || !x || !y) return -1;
    if (a->m != x->n || a->n != y->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu) return -1;
    return metal_ger(a, alpha, x, y);
}

/* ================================================================
 * pfr_*: primitiva neuralium retium — conatur GPU, cadit ad CPU
 * ================================================================ */

int pfr_rmsnorm_f(pfr_vector_f_t *o, const pfr_vector_f_t *x,
                  const pfr_vector_f_t *w, float eps)
{
    if (!o || !x || !w) return -1;
    if (mc.usar_gpu && o->gpu && x->gpu && w->gpu)
        return metal_rmsnorm(o->gpu, x->gpu, w->gpu, x->n, eps);
    return pfr_cpu_rmsnorm_f(o, x, w, eps);
}

int pfr_swiglu_f(pfr_vector_f_t *o, const pfr_vector_f_t *a,
                 const pfr_vector_f_t *b)
{
    if (!o || !a || !b) return -1;
    if (mc.usar_gpu && o->gpu && a->gpu && b->gpu)
        return metal_swiglu(o->gpu, a->gpu, b->gpu, a->n);
    return pfr_cpu_swiglu_f(o, a, b);
}

int pfr_softmax_f(pfr_vector_f_t *x)
{
    if (!x) return -1;
    if (mc.usar_gpu && x->gpu)
        return metal_softmax(x->gpu, x->n);
    return pfr_cpu_softmax_f(x);
}

int pfr_rope_f(pfr_vector_f_t *v, int positio)
{
    if (!v) return -1;
    if (mc.usar_gpu && v->gpu)
        return metal_rope(v->gpu, v->n, positio);
    return pfr_cpu_rope_f(v, positio);
}

/* pfr_gpu_*: primitiva neuralium retium — GPU solum */

int pfr_gpu_rmsnorm_f(pfr_vector_f_t *o, const pfr_vector_f_t *x,
                      const pfr_vector_f_t *w, float eps)
{
    if (!o || !x || !w) return -1;
    if (!mc.usar_gpu || !o->gpu || !x->gpu || !w->gpu) return -1;
    return metal_rmsnorm(o->gpu, x->gpu, w->gpu, x->n, eps);
}

int pfr_gpu_swiglu_f(pfr_vector_f_t *o, const pfr_vector_f_t *a,
                     const pfr_vector_f_t *b)
{
    if (!o || !a || !b) return -1;
    if (!mc.usar_gpu || !o->gpu || !a->gpu || !b->gpu) return -1;
    return metal_swiglu(o->gpu, a->gpu, b->gpu, a->n);
}

int pfr_gpu_softmax_f(pfr_vector_f_t *x)
{
    if (!x) return -1;
    if (!mc.usar_gpu || !x->gpu) return -1;
    return metal_softmax(x->gpu, x->n);
}

int pfr_gpu_rope_f(pfr_vector_f_t *v, int positio)
{
    if (!v) return -1;
    if (!mc.usar_gpu || !v->gpu) return -1;
    return metal_rope(v->gpu, v->n, positio);
}

/* gpu_*_d — semper fallit: Metal non sustinet double in GPU */
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
