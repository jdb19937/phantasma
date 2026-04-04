/*
 * computo_cpu.c — operationes BLAS in CPU
 * ==============================================
 *
 * Functiones pfr_cpu_*_f et pfr_cpu_*_d — includuntur ab computo_plat.c
 * et computo_cuda.cu.  Non compilatur seorsum.
 * Parens plica debet includere computo.h, <stdlib.h>, et <math.h>.
 */

/* ================================================================
 * GEMM: C = A * B (float)
 * ================================================================ */

int pfr_cpu_matmat_f(
    pfr_matrix_f_t *c,
    const pfr_matrix_f_t *a, const pfr_matrix_f_t *b
) {
    if (!c || !a || !b)
        return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n)
        return -1;
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
 * GEMV: y = A * x (float)
 * ================================================================ */

int pfr_cpu_matvec_f(
    pfr_vector_f_t *y,
    const pfr_matrix_f_t *a, const pfr_vector_f_t *x
) {
    if (!y || !a || !x)
        return -1;
    if (a->n != x->n || y->n != a->m)
        return -1;
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
 * DOT: *res = x . y (float)
 * ================================================================ */

int pfr_cpu_dotum_f(
    float *res,
    const pfr_vector_f_t *x, const pfr_vector_f_t *y
) {
    if (!res || !x || !y || x->n != y->n)
        return -1;
    float s = 0.0f;
    for (int i = 0; i < x->n; i++)
        s += x->data[i] * y->data[i];
    *res = s;
    return 0;
}

/* ================================================================
 * SCAL: x = alpha * x (float)
 * ================================================================ */

int pfr_cpu_scalare_f(pfr_vector_f_t *x, float alpha)
{
    if (!x)
        return -1;
    for (int i = 0; i < x->n; i++)
        x->data[i] *= alpha;
    return 0;
}

/* ================================================================
 * AXPY: y = alpha * x + y (float)
 * ================================================================ */

int pfr_cpu_axpy_f(pfr_vector_f_t *y, float alpha, const pfr_vector_f_t *x)
{
    if (!y || !x || y->n != x->n)
        return -1;
    for (int i = 0; i < y->n; i++)
        y->data[i] += alpha * x->data[i];
    return 0;
}

/* ================================================================
 * GEMV^T: y = A^T * x (float)
 * ================================================================ */

int pfr_cpu_matvec_trans_f(
    pfr_vector_f_t *y,
    const pfr_matrix_f_t *a, const pfr_vector_f_t *x
) {
    if (!y || !a || !x)
        return -1;
    if (a->m != x->n || y->n != a->n)
        return -1;
    int m = a->m, n = a->n;
    for (int j = 0; j < n; j++)
        y->data[j] = 0.0f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            y->data[j] += a->data[i*n + j] * x->data[i];
    return 0;
}

/* ================================================================
 * GER: A += alpha * x * y^T (float)
 * ================================================================ */

int pfr_cpu_ger_f(
    pfr_matrix_f_t *a, float alpha,
    const pfr_vector_f_t *x, const pfr_vector_f_t *y
) {
    if (!a || !x || !y)
        return -1;
    if (a->m != x->n || a->n != y->n)
        return -1;
    int m = a->m, n = a->n;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a->data[i*n + j] += alpha * x->data[i] * y->data[j];
    return 0;
}

/* ================================================================
 * GEMM: C = A * B (double)
 * ================================================================ */

int pfr_cpu_matmat_d(
    pfr_matrix_d_t *c,
    const pfr_matrix_d_t *a, const pfr_matrix_d_t *b
) {
    if (!c || !a || !b)
        return -1;
    if (a->n != b->m || c->m != a->m || c->n != b->n)
        return -1;
    int m = a->m, k = a->n, n = b->n;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int l = 0; l < k; l++)
            s += a->data[i*k + l] * b->data[l*n + j];
        c->data[i*n + j] = s;
    }
    return 0;
}

/* ================================================================
 * GEMV: y = A * x (double)
 * ================================================================ */

int pfr_cpu_matvec_d(
    pfr_vector_d_t *y,
    const pfr_matrix_d_t *a, const pfr_vector_d_t *x
) {
    if (!y || !a || !x)
        return -1;
    if (a->n != x->n || y->n != a->m)
        return -1;
    int m = a->m, n = a->n;
    for (int i = 0; i < m; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++)
            s += a->data[i*n + j] * x->data[j];
        y->data[i] = s;
    }
    return 0;
}

/* ================================================================
 * DOT: *res = x . y (double)
 * ================================================================ */

int pfr_cpu_dotum_d(
    double *res,
    const pfr_vector_d_t *x, const pfr_vector_d_t *y
) {
    if (!res || !x || !y || x->n != y->n)
        return -1;
    double s = 0.0;
    for (int i = 0; i < x->n; i++)
        s += x->data[i] * y->data[i];
    *res = s;
    return 0;
}

/* ================================================================
 * SCAL: x = alpha * x (double)
 * ================================================================ */

int pfr_cpu_scalare_d(pfr_vector_d_t *x, double alpha)
{
    if (!x)
        return -1;
    for (int i = 0; i < x->n; i++)
        x->data[i] *= alpha;
    return 0;
}

/* ================================================================
 * AXPY: y = alpha * x + y (double)
 * ================================================================ */

int pfr_cpu_axpy_d(pfr_vector_d_t *y, double alpha, const pfr_vector_d_t *x)
{
    if (!y || !x || y->n != x->n)
        return -1;
    for (int i = 0; i < y->n; i++)
        y->data[i] += alpha * x->data[i];
    return 0;
}

/* ================================================================
 * RMSNorm: o[j] = w[j] * x[j] * rsqrt(mean(x^2) + eps) (float)
 * ================================================================ */

int pfr_cpu_rmsnorm_f(
    pfr_vector_f_t *o, const pfr_vector_f_t *x,
    const pfr_vector_f_t *w, float eps
) {
    if (!o || !x || !w)
        return -1;
    int n = x->n;
    if (o->n != n || w->n != n)
        return -1;
    float ss = 0.0f;
    for (int j = 0; j < n; j++)
        ss += x->data[j] * x->data[j];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int j = 0; j < n; j++)
        o->data[j] = w->data[j] * ss * x->data[j];
    return 0;
}

/* ================================================================
 * SwiGLU: o[i] = silu(a[i]) * b[i] (float)
 * ================================================================ */

int pfr_cpu_swiglu_f(
    pfr_vector_f_t *o, const pfr_vector_f_t *a,
    const pfr_vector_f_t *b
) {
    if (!o || !a || !b)
        return -1;
    int n = a->n;
    if (o->n != n || b->n != n)
        return -1;
    for (int i = 0; i < n; i++) {
        float sig  = 1.0f / (1.0f + expf(-a->data[i]));
        o->data[i] = a->data[i] * sig * b->data[i];
    }
    return 0;
}

/* ================================================================
 * Softmax in situ (float)
 * ================================================================ */

int pfr_cpu_softmax_f(pfr_vector_f_t *x)
{
    if (!x || x->n <= 0)
        return -1;
    int n    = x->n;
    float mx = x->data[0];
    for (int i = 1; i < n; i++)
        if (x->data[i] > mx)
            mx = x->data[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        x->data[i] = expf(x->data[i] - mx);
        s += x->data[i];
    }
    for (int i = 0; i < n; i++)
        x->data[i] /= s;
    return 0;
}

/* ================================================================
 * RoPE: rotatio per positionem (float)
 * ================================================================ */

int pfr_cpu_rope_f(pfr_vector_f_t *v, int positio)
{
    if (!v || v->n <= 0)
        return -1;
    int n = v->n;
    for (int i = 0; i < n; i += 2) {
        float freq     = 1.0f / powf(10000.0f, (float)i / n);
        float val      = positio * freq;
        float fcr      = cosf(val), fci = sinf(val);
        float v0       = v->data[i], v1 = v->data[i + 1];
        v->data[i]     = v0 * fcr - v1 * fci;
        v->data[i + 1] = v0 * fci + v1 * fcr;
    }
    return 0;
}

/* ================================================================
 * Attentio causalis multi-capitis (float, raw pointers)
 * ================================================================ */

int pfr_cpu_attentio_f(
    float *o, const float *q,
    const float *cache_k, const float *cache_v,
    float *att,
    int d, int n_capita, int n_capita_kv,
    int positio, int longitudo_max
) {
    if (!o || !q || !cache_k || !cache_v || !att)
        return -1;
    if (d <= 0 || n_capita <= 0 || n_capita_kv <= 0)
        return -1;
    int hd      = d / n_capita;
    int kv_dim  = hd * n_capita_kv;
    int kv_mul  = n_capita / n_capita_kv;
    float inv_s = 1.0f / sqrtf((float)hd);

    memset(o, 0, (size_t)d * sizeof(float));

    for (int h = 0; h < n_capita; h++) {
        const float *q_h = q + h * hd;
        float *att_h     = att + h * longitudo_max;
        float *o_h       = o + h * hd;
        int hkv          = h / kv_mul;

        for (int t = 0; t <= positio; t++) {
            const float *k_t = cache_k + (size_t)t * kv_dim + hkv * hd;
            float sc         = 0.0f;
            for (int i = 0; i < hd; i++)
                sc += q_h[i] * k_t[i];
            att_h[t] = sc * inv_s;
        }

        pfr_vector_f_t att_v = { positio + 1, att_h, NULL };
        pfr_cpu_softmax_f(&att_v);

        for (int t = 0; t <= positio; t++) {
            const float *v_t = cache_v + (size_t)t * kv_dim + hkv * hd;
            float a = att_h[t];
            for (int i = 0; i < hd; i++)
                o_h[i] += a * v_t[i];
        }
    }
    return 0;
}
