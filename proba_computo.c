/*
 * proba_computo.c — probationes computationis GPU/CPU
 * =====================================================
 *
 * Probat pfr_matmat, pfr_matvec, pfr_dotum, pfr_scalare, pfr_axpy
 * in variante _f (float) et _d (double), in GPU (si adest) et CPU.
 */

#include "computo.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ================================================================
 * auxiliaria
 * ================================================================ */

static int probat(const char *nomen, int ok)
{
    printf("%s: %s\n", nomen, ok ? "PROBA" : "DEFECIT");
    return ok ? 0 : 1;
}

static int fere_aequalis_f(float a, float b)
{
    return fabsf(a - b) < 1e-4f;
}

static int fere_aequalis_d(double a, double b)
{
    return fabs(a - b) < 1e-10;
}

/* ================================================================
 * probationes float
 * ================================================================ */

/* matmat: I_3 * A = A */
static int proba_matmat_f(void)
{
    pfr_matrix_f_t *a = pfr_matrix_crea_f(3, 3);
    pfr_matrix_f_t *b = pfr_matrix_crea_f(3, 3);
    pfr_matrix_f_t *c = pfr_matrix_crea_f(3, 3);

    /* A = identitas */
    a->data[0] = 1;
    a->data[4] = 1;
    a->data[8] = 1;
    /* B = [[1,2,3],[4,5,6],[7,8,9]] */
    for (int i = 0; i < 9; i++)
        b->data[i] = (float)(i + 1);

    pfr_in_gpu_mitte_f(a);
    pfr_in_gpu_mitte_f(b);
    pfr_in_gpu_mitte_f(c);
    pfr_matmat_f(c, a, b);
    pfr_ex_gpu_cape_f(c);

    int ok = 1;
    for (int i = 0; i < 9; i++)
        if (!fere_aequalis_f(c->data[i], b->data[i])) {
        ok = 0;
        break;
    }

    pfr_matrix_destrue_f(a);
    pfr_matrix_destrue_f(b);
    pfr_matrix_destrue_f(c);
    return probat("matmat_f (I*B=B)", ok);
}

/* matvec: I_3 * x = x */
static int proba_matvec_f(void)
{
    pfr_matrix_f_t *a = pfr_matrix_crea_f(3, 3);
    pfr_vector_f_t *x = pfr_vector_crea_f(3);
    pfr_vector_f_t *y = pfr_vector_crea_f(3);

    a->data[0] = 1;
    a->data[4] = 1;
    a->data[8] = 1;
    x->data[0] = 2;
    x->data[1] = 5;
    x->data[2] = 7;

    pfr_in_gpu_mitte_f(a);
    pfr_in_gpu_mitte_vf(x);
    pfr_in_gpu_mitte_vf(y);
    pfr_matvec_f(y, a, x);
    pfr_ex_gpu_cape_vf(y);

    int ok = fere_aequalis_f(y->data[0], 2.0f)
        && fere_aequalis_f(y->data[1], 5.0f)
        && fere_aequalis_f(y->data[2], 7.0f);

    pfr_matrix_destrue_f(a);
    pfr_vector_destrue_f(x);
    pfr_vector_destrue_f(y);
    return probat("matvec_f (I*x=x)", ok);
}

/* dotum: [1,2,3] . [4,5,6] = 32 */
static int proba_dotum_f(void)
{
    pfr_vector_f_t *x = pfr_vector_crea_f(3);
    pfr_vector_f_t *y = pfr_vector_crea_f(3);
    x->data[0]        = 1;
    x->data[1]        = 2;
    x->data[2]        = 3;
    y->data[0]        = 4;
    y->data[1]        = 5;
    y->data[2]        = 6;
    pfr_in_gpu_mitte_vf(x);
    pfr_in_gpu_mitte_vf(y);

    float res = 0.0f;
    pfr_dotum_f(&res, x, y);

    pfr_vector_destrue_f(x);
    pfr_vector_destrue_f(y);
    return probat("dotum_f ([1,2,3].[4,5,6]=32)", fere_aequalis_f(res, 32.0f));
}

/* scalare: [1,2,3] * 3 = [3,6,9] */
static int proba_scalare_f(void)
{
    pfr_vector_f_t *x = pfr_vector_crea_f(3);
    x->data[0]        = 1;
    x->data[1]        = 2;
    x->data[2]        = 3;
    pfr_in_gpu_mitte_vf(x);
    pfr_scalare_f(x, 3.0f);
    pfr_ex_gpu_cape_vf(x);

    int ok = fere_aequalis_f(x->data[0], 3.0f)
        && fere_aequalis_f(x->data[1], 6.0f)
        && fere_aequalis_f(x->data[2], 9.0f);

    pfr_vector_destrue_f(x);
    return probat("scalare_f ([1,2,3]*3=[3,6,9])", ok);
}

/* axpy: 2*[1,1,1] + [1,1,1] = [3,3,3] */
static int proba_axpy_f(void)
{
    pfr_vector_f_t *x = pfr_vector_crea_f(3);
    pfr_vector_f_t *y = pfr_vector_crea_f(3);
    x->data[0]        = x->data[1] = x->data[2] = 1.0f;
    y->data[0]        = y->data[1] = y->data[2] = 1.0f;
    pfr_in_gpu_mitte_vf(x);
    pfr_in_gpu_mitte_vf(y);
    pfr_axpy_f(y, 2.0f, x);
    pfr_ex_gpu_cape_vf(y);

    int ok = fere_aequalis_f(y->data[0], 3.0f)
        && fere_aequalis_f(y->data[1], 3.0f)
        && fere_aequalis_f(y->data[2], 3.0f);

    pfr_vector_destrue_f(x);
    pfr_vector_destrue_f(y);
    return probat("axpy_f (2*[1,1,1]+[1,1,1]=[3,3,3])", ok);
}

/* matmat magna: A*A pro A=[[2,1],[1,2]] */
static int proba_matmat_magna_f(void)
{
    /* A = [[2,1],[1,2]]
     * A*A = [[5,4],[4,5]] */
    pfr_matrix_f_t *a = pfr_matrix_crea_f(2, 2);
    pfr_matrix_f_t *c = pfr_matrix_crea_f(2, 2);
    a->data[0]        = 2;
    a->data[1]        = 1;
    a->data[2]        = 1;
    a->data[3]        = 2;
    pfr_in_gpu_mitte_f(a);
    pfr_in_gpu_mitte_f(c);
    pfr_matmat_f(c, a, a);
    pfr_ex_gpu_cape_f(c);

    int ok = fere_aequalis_f(c->data[0], 5.0f)
        && fere_aequalis_f(c->data[1], 4.0f)
        && fere_aequalis_f(c->data[2], 4.0f)
        && fere_aequalis_f(c->data[3], 5.0f);

    pfr_matrix_destrue_f(a);
    pfr_matrix_destrue_f(c);
    return probat("matmat_f ([[2,1],[1,2]]^2=[[5,4],[4,5]])", ok);
}

/* ================================================================
 * probationes double
 * ================================================================ */

/* matmat: I_3 * A = A */
static int proba_matmat_d(void)
{
    pfr_matrix_d_t *a = pfr_matrix_crea_d(3, 3);
    pfr_matrix_d_t *b = pfr_matrix_crea_d(3, 3);
    pfr_matrix_d_t *c = pfr_matrix_crea_d(3, 3);

    a->data[0] = 1;
    a->data[4] = 1;
    a->data[8] = 1;
    for (int i = 0; i < 9; i++)
        b->data[i] = (double)(i + 1);

    pfr_in_gpu_mitte_d(a);
    pfr_in_gpu_mitte_d(b);
    pfr_in_gpu_mitte_d(c);
    pfr_matmat_d(c, a, b);
    pfr_ex_gpu_cape_d(c);

    int ok = 1;
    for (int i = 0; i < 9; i++)
        if (!fere_aequalis_d(c->data[i], b->data[i])) {
        ok = 0;
        break;
    }

    pfr_matrix_destrue_d(a);
    pfr_matrix_destrue_d(b);
    pfr_matrix_destrue_d(c);
    return probat("matmat_d (I*B=B)", ok);
}

/* matvec: I_3 * x = x */
static int proba_matvec_d(void)
{
    pfr_matrix_d_t *a = pfr_matrix_crea_d(3, 3);
    pfr_vector_d_t *x = pfr_vector_crea_d(3);
    pfr_vector_d_t *y = pfr_vector_crea_d(3);

    a->data[0] = 1;
    a->data[4] = 1;
    a->data[8] = 1;
    x->data[0] = 2;
    x->data[1] = 5;
    x->data[2] = 7;

    pfr_in_gpu_mitte_d(a);
    pfr_in_gpu_mitte_vd(x);
    pfr_in_gpu_mitte_vd(y);
    pfr_matvec_d(y, a, x);
    pfr_ex_gpu_cape_vd(y);

    int ok = fere_aequalis_d(y->data[0], 2.0)
        && fere_aequalis_d(y->data[1], 5.0)
        && fere_aequalis_d(y->data[2], 7.0);

    pfr_matrix_destrue_d(a);
    pfr_vector_destrue_d(x);
    pfr_vector_destrue_d(y);
    return probat("matvec_d (I*x=x)", ok);
}

/* dotum: [1,2,3] . [4,5,6] = 32 */
static int proba_dotum_d(void)
{
    pfr_vector_d_t *x = pfr_vector_crea_d(3);
    pfr_vector_d_t *y = pfr_vector_crea_d(3);
    x->data[0]        = 1;
    x->data[1]        = 2;
    x->data[2]        = 3;
    y->data[0]        = 4;
    y->data[1]        = 5;
    y->data[2]        = 6;
    pfr_in_gpu_mitte_vd(x);
    pfr_in_gpu_mitte_vd(y);

    double res = 0.0;
    pfr_dotum_d(&res, x, y);

    pfr_vector_destrue_d(x);
    pfr_vector_destrue_d(y);
    return probat("dotum_d ([1,2,3].[4,5,6]=32)", fere_aequalis_d(res, 32.0));
}

/* scalare: [1,2,3] * 3 = [3,6,9] */
static int proba_scalare_d(void)
{
    pfr_vector_d_t *x = pfr_vector_crea_d(3);
    x->data[0]        = 1;
    x->data[1]        = 2;
    x->data[2]        = 3;
    pfr_in_gpu_mitte_vd(x);
    pfr_scalare_d(x, 3.0);
    pfr_ex_gpu_cape_vd(x);

    int ok = fere_aequalis_d(x->data[0], 3.0)
        && fere_aequalis_d(x->data[1], 6.0)
        && fere_aequalis_d(x->data[2], 9.0);

    pfr_vector_destrue_d(x);
    return probat("scalare_d ([1,2,3]*3=[3,6,9])", ok);
}

/* axpy: 2*[1,1,1] + [1,1,1] = [3,3,3] */
static int proba_axpy_d(void)
{
    pfr_vector_d_t *x = pfr_vector_crea_d(3);
    pfr_vector_d_t *y = pfr_vector_crea_d(3);
    x->data[0]        = x->data[1] = x->data[2] = 1.0;
    y->data[0]        = y->data[1] = y->data[2] = 1.0;
    pfr_in_gpu_mitte_vd(x);
    pfr_in_gpu_mitte_vd(y);
    pfr_axpy_d(y, 2.0, x);
    pfr_ex_gpu_cape_vd(y);

    int ok = fere_aequalis_d(y->data[0], 3.0)
        && fere_aequalis_d(y->data[1], 3.0)
        && fere_aequalis_d(y->data[2], 3.0);

    pfr_vector_destrue_d(x);
    pfr_vector_destrue_d(y);
    return probat("axpy_d (2*[1,1,1]+[1,1,1]=[3,3,3])", ok);
}

/* matmat magna: A*A pro A=[[2,1],[1,2]] */
static int proba_matmat_magna_d(void)
{
    pfr_matrix_d_t *a = pfr_matrix_crea_d(2, 2);
    pfr_matrix_d_t *c = pfr_matrix_crea_d(2, 2);
    a->data[0]        = 2;
    a->data[1]        = 1;
    a->data[2]        = 1;
    a->data[3]        = 2;
    pfr_in_gpu_mitte_d(a);
    pfr_in_gpu_mitte_d(c);
    pfr_matmat_d(c, a, a);
    pfr_ex_gpu_cape_d(c);

    int ok = fere_aequalis_d(c->data[0], 5.0)
        && fere_aequalis_d(c->data[1], 4.0)
        && fere_aequalis_d(c->data[2], 4.0)
        && fere_aequalis_d(c->data[3], 5.0);

    pfr_matrix_destrue_d(a);
    pfr_matrix_destrue_d(c);
    return probat("matmat_d ([[2,1],[1,2]]^2=[[5,4],[4,5]])", ok);
}

/* ================================================================
 * principale
 * ================================================================ */

int main(void)
{
    int modus = pfr_computo_initia();
    printf(
        "pfr_computo: %s\n\n",
        modus == 0 ? "GPU" : "CPU fallback"
    );

    int defecta = 0;

    printf("--- float ---\n");
    defecta += proba_matmat_f();
    defecta += proba_matmat_magna_f();
    defecta += proba_matvec_f();
    defecta += proba_dotum_f();
    defecta += proba_scalare_f();
    defecta += proba_axpy_f();

    printf("\n--- double ---\n");
    defecta += proba_matmat_d();
    defecta += proba_matmat_magna_d();
    defecta += proba_matvec_d();
    defecta += proba_dotum_d();
    defecta += proba_scalare_d();
    defecta += proba_axpy_d();

    printf(
        "\n%s (%d defecta)\n",
        defecta == 0 ? "OMNIA PROBATA" : "ALIQUA DEFECERUNT",
        defecta
    );

    pfr_computo_fini();
    return defecta ? 1 : 0;
}
