/*
 * proba_computo.c — probationes computationis GPU/CPU
 * =====================================================
 *
 * Probat pfr_matmat, pfr_matvec, pfr_dotum, pfr_scalare, pfr_axpy
 * in GPU (si adest) et CPU.
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

static int fere_aequalis(float a, float b)
{
    return fabsf(a - b) < 1e-4f;
}

/* ================================================================
 * probationes
 * ================================================================ */

/* matmat: I_3 * A = A */
static int proba_matmat(void)
{
    pfr_matrix_t *a = pfr_matrix_crea(3, 3);
    pfr_matrix_t *b = pfr_matrix_crea(3, 3);
    pfr_matrix_t *c = pfr_matrix_crea(3, 3);

    /* A = identitas */
    a->data[0] = 1; a->data[4] = 1; a->data[8] = 1;
    /* B = [[1,2,3],[4,5,6],[7,8,9]] */
    for (int i = 0; i < 9; i++) b->data[i] = (float)(i + 1);

    pfr_in_gpu_mitte(a); pfr_in_gpu_mitte(b); pfr_in_gpu_mitte(c);
    pfr_matmat(c, a, b);
    pfr_ex_gpu_cape(c);

    int ok = 1;
    for (int i = 0; i < 9; i++)
        if (!fere_aequalis(c->data[i], b->data[i])) { ok = 0; break; }

    pfr_matrix_destrue(a); pfr_matrix_destrue(b); pfr_matrix_destrue(c);
    return probat("matmat (I*B=B)", ok);
}

/* matvec: I_3 * x = x */
static int proba_matvec(void)
{
    pfr_matrix_t *a = pfr_matrix_crea(3, 3);
    pfr_vector_t *x = pfr_vector_crea(3);
    pfr_vector_t *y = pfr_vector_crea(3);

    a->data[0] = 1; a->data[4] = 1; a->data[8] = 1;
    x->data[0] = 2; x->data[1] = 5; x->data[2] = 7;

    pfr_in_gpu_mitte(a); pfr_in_gpu_mitte_v(x); pfr_in_gpu_mitte_v(y);
    pfr_matvec(y, a, x);
    pfr_ex_gpu_cape_v(y);

    int ok = fere_aequalis(y->data[0], 2.0f)
          && fere_aequalis(y->data[1], 5.0f)
          && fere_aequalis(y->data[2], 7.0f);

    pfr_matrix_destrue(a); pfr_vector_destrue(x); pfr_vector_destrue(y);
    return probat("matvec (I*x=x)", ok);
}

/* dotum: [1,2,3] . [4,5,6] = 32 */
static int proba_dotum(void)
{
    pfr_vector_t *x = pfr_vector_crea(3);
    pfr_vector_t *y = pfr_vector_crea(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    y->data[0] = 4; y->data[1] = 5; y->data[2] = 6;
    pfr_in_gpu_mitte_v(x); pfr_in_gpu_mitte_v(y);

    float res = 0.0f;
    pfr_dotum(&res, x, y);

    pfr_vector_destrue(x); pfr_vector_destrue(y);
    return probat("dotum ([1,2,3].[4,5,6]=32)", fere_aequalis(res, 32.0f));
}

/* scalare: [1,2,3] * 3 = [3,6,9] */
static int proba_scalare(void)
{
    pfr_vector_t *x = pfr_vector_crea(3);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    pfr_in_gpu_mitte_v(x);
    pfr_scalare(x, 3.0f);
    pfr_ex_gpu_cape_v(x);

    int ok = fere_aequalis(x->data[0], 3.0f)
          && fere_aequalis(x->data[1], 6.0f)
          && fere_aequalis(x->data[2], 9.0f);

    pfr_vector_destrue(x);
    return probat("scalare ([1,2,3]*3=[3,6,9])", ok);
}

/* axpy: 2*[1,1,1] + [1,1,1] = [3,3,3] */
static int proba_axpy(void)
{
    pfr_vector_t *x = pfr_vector_crea(3);
    pfr_vector_t *y = pfr_vector_crea(3);
    x->data[0] = x->data[1] = x->data[2] = 1.0f;
    y->data[0] = y->data[1] = y->data[2] = 1.0f;
    pfr_in_gpu_mitte_v(x); pfr_in_gpu_mitte_v(y);
    pfr_axpy(y, 2.0f, x);
    pfr_ex_gpu_cape_v(y);

    int ok = fere_aequalis(y->data[0], 3.0f)
          && fere_aequalis(y->data[1], 3.0f)
          && fere_aequalis(y->data[2], 3.0f);

    pfr_vector_destrue(x); pfr_vector_destrue(y);
    return probat("axpy (2*[1,1,1]+[1,1,1]=[3,3,3])", ok);
}

/* matmat magna: A*A pro A=[[2,1],[1,2]] */
static int proba_matmat_magna(void)
{
    /* A = [[2,1],[1,2]]
     * A*A = [[5,4],[4,5]] */
    pfr_matrix_t *a = pfr_matrix_crea(2, 2);
    pfr_matrix_t *c = pfr_matrix_crea(2, 2);
    a->data[0] = 2; a->data[1] = 1;
    a->data[2] = 1; a->data[3] = 2;
    pfr_in_gpu_mitte(a); pfr_in_gpu_mitte(c);
    pfr_matmat(c, a, a);
    pfr_ex_gpu_cape(c);

    int ok = fere_aequalis(c->data[0], 5.0f)
          && fere_aequalis(c->data[1], 4.0f)
          && fere_aequalis(c->data[2], 4.0f)
          && fere_aequalis(c->data[3], 5.0f);

    pfr_matrix_destrue(a); pfr_matrix_destrue(c);
    return probat("matmat ([[2,1],[1,2]]^2=[[5,4],[4,5]])", ok);
}

/* ================================================================
 * principale
 * ================================================================ */

int main(void)
{
    int modus = pfr_computo_initia();
    printf("pfr_computo: %s\n\n",
           modus == 0 ? "GPU" : "CPU fallback");

    int defecta = 0;
    defecta += proba_matmat();
    defecta += proba_matmat_magna();
    defecta += proba_matvec();
    defecta += proba_dotum();
    defecta += proba_scalare();
    defecta += proba_axpy();

    printf("\n%s (%d defecta)\n",
           defecta == 0 ? "OMNIA PROBATA" : "ALIQUA DEFECERUNT",
           defecta);

    pfr_computo_fini();
    return defecta ? 1 : 0;
}
