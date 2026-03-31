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
    "}\n";

/* ================================================================
 * status globalis
 * ================================================================ */

#define MC_MATMAT  0
#define MC_MATVEC  1
#define MC_DOTUM   2
#define MC_SCALARE 3
#define MC_AXPY    4
#define MC_N_PLENA 5

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
            "kern_scalare", "kern_axpy"
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
 * crea / destrue
 * ================================================================ */

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
    if (a->gpu) { [mc_id_buffer(a->gpu) release]; a->gpu = NULL; }
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
    if (v->gpu) { [mc_id_buffer(v->gpu) release]; v->gpu = NULL; }
    free(v->data);
    free(v);
}

/* ================================================================
 * translatio CPU <-> GPU
 * ================================================================ */

int pfr_in_gpu_mitte(pfr_matrix_t *a)
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

int pfr_ex_gpu_cape(pfr_matrix_t *a)
{
    if (!a || !a->gpu) return 0;
    size_t bytes = (size_t)a->m * a->n * sizeof(float);
    memcpy(a->data, [mc_id_buffer(a->gpu) contents], bytes);
    return 0;
}

int pfr_in_gpu_mitte_v(pfr_vector_t *v)
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

int pfr_ex_gpu_cape_v(pfr_vector_t *v)
{
    if (!v || !v->gpu) return 0;
    size_t bytes = (size_t)v->n * sizeof(float);
    memcpy(v->data, [mc_id_buffer(v->gpu) contents], bytes);
    return 0;
}

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

static int metal_matmat(pfr_matrix_t *c,
                         const pfr_matrix_t *a, const pfr_matrix_t *b)
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

static int metal_matvec(pfr_vector_t *y,
                         const pfr_matrix_t *a, const pfr_vector_t *x)
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
                        const pfr_vector_t *x, const pfr_vector_t *y)
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

static int metal_scalare(pfr_vector_t *x, float alpha)
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

static int metal_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
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

/* ================================================================
 * pfr_* — conatur GPU; cadit ad CPU si non adest
 * ================================================================ */

int pfr_matmat(pfr_matrix_t *c,
               const pfr_matrix_t *a, const pfr_matrix_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !b->gpu || !c->gpu)
        return pfr_cpu_matmat(c, a, b);
    return metal_matmat(c, a, b);
}

int pfr_matvec(pfr_vector_t *y,
               const pfr_matrix_t *a, const pfr_vector_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu)
        return pfr_cpu_matvec(y, a, x);
    return metal_matvec(y, a, x);
}

int pfr_dotum(float *res, const pfr_vector_t *x, const pfr_vector_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!mc.usar_gpu || !x->gpu || !y->gpu)
        return pfr_cpu_dotum(res, x, y);
    return metal_dotum(res, x, y);
}

int pfr_scalare(pfr_vector_t *x, float alpha)
{
    if (!x) return -1;
    if (!mc.usar_gpu || !x->gpu) return pfr_cpu_scalare(x, alpha);
    return metal_scalare(x, alpha);
}

int pfr_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!mc.usar_gpu || !y->gpu || !x->gpu) return pfr_cpu_axpy(y, alpha, x);
    return metal_axpy(y, alpha, x);
}

/* ================================================================
 * gpu_* — fallit si GPU non adest vel data non in GPU
 * ================================================================ */

int pfr_gpu_matmat(pfr_matrix_t *c,
               const pfr_matrix_t *a, const pfr_matrix_t *b)
{
    if (!c || !a || !b) return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n) return -1;
    if (!mc.usar_gpu || !a->gpu || !b->gpu || !c->gpu) return -1;
    return metal_matmat(c, a, b);
}

int pfr_gpu_matvec(pfr_vector_t *y,
               const pfr_matrix_t *a, const pfr_vector_t *x)
{
    if (!y || !a || !x) return -1;
    if (a->n != x->n || y->n != a->m) return -1;
    if (!mc.usar_gpu || !a->gpu || !x->gpu || !y->gpu) return -1;
    return metal_matvec(y, a, x);
}

int pfr_gpu_dotum(float *res, const pfr_vector_t *x, const pfr_vector_t *y)
{
    if (!res || !x || !y || x->n != y->n) return -1;
    if (!mc.usar_gpu || !x->gpu || !y->gpu) return -1;
    return metal_dotum(res, x, y);
}

int pfr_gpu_scalare(pfr_vector_t *x, float alpha)
{
    if (!x || !mc.usar_gpu || !x->gpu) return -1;
    return metal_scalare(x, alpha);
}

int pfr_gpu_axpy(pfr_vector_t *y, float alpha, const pfr_vector_t *x)
{
    if (!y || !x || y->n != x->n) return -1;
    if (!mc.usar_gpu || !y->gpu || !x->gpu) return -1;
    return metal_axpy(y, alpha, x);
}
