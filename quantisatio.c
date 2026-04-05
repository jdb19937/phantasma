/*
 * quantisatio.c — algorithmi quantisationis colorum
 * ===================================================
 *
 * RGB pixelos in paletam finitam reducit.
 *
 * Modi:
 *   1. Median-cut  — capsas per dimensionem longissimam scindit
 *   2. Octarboris  — arborem octonariam aedificat et folia fundit
 *   3. K-media     — median-cut semine utitur, centroides iterative optimat
 *
 * Omnes modi histogrammo 32x32x32 utuntur pro velocitate.
 */

#include "quantisatio.h"

#include <stdlib.h>
#include <string.h>

/* ================================================================
 * histogrammum commune
 * ================================================================ */

#define HIST_BITS  5
#define HIST_DIM   (1 << HIST_BITS)   /* 32 */
#define HIST_SHIFT (8 - HIST_BITS)    /* 3 */
#define HIST_N     (HIST_DIM * HIST_DIM * HIST_DIM) /* 32768 */

typedef struct {
    int32_t numerus;
    int64_t r_sum, g_sum, b_sum;
} hist_cella_t;

static void histogrammum_aedifica(
    hist_cella_t *hist, const uint8_t *rgb, int n_pix
) {
    memset(hist, 0, HIST_N * sizeof(hist_cella_t));
    for (int i = 0; i < n_pix; i++) {
        int r  = rgb[i * 3 + 0];
        int g  = rgb[i * 3 + 1];
        int b  = rgb[i * 3 + 2];
        int hi = (r >> HIST_SHIFT) * HIST_DIM * HIST_DIM +
                 (g >> HIST_SHIFT) * HIST_DIM +
                 (b >> HIST_SHIFT);
        hist[hi].numerus++;
        hist[hi].r_sum += r;
        hist[hi].g_sum += g;
        hist[hi].b_sum += b;
    }
}

/* ================================================================
 * median-cut
 * ================================================================ */

typedef struct {
    int r_min, r_max;
    int g_min, g_max;
    int b_min, b_max;
    int64_t r_sum, g_sum, b_sum;
    int numerus;
} capsa_t;

static int capsa_amplitudo(const capsa_t *c)
{
    int dr = c->r_max - c->r_min;
    int dg = c->g_max - c->g_min;
    int db = c->b_max - c->b_min;
    if (dr >= dg && dr >= db)
        return 0;
    if (dg >= dr && dg >= db)
        return 1;
    return 2;
}

static int capsa_magnitudo(const capsa_t *c)
{
    int dr = c->r_max - c->r_min;
    int dg = c->g_max - c->g_min;
    int db = c->b_max - c->b_min;
    if (dr >= dg && dr >= db)
        return dr;
    if (dg >= dr && dg >= db)
        return dg;
    return db;
}

void paletam_genera(
    const uint8_t *rgb, int n_pix,
    uint8_t paleta[][3], int n_colorum
) {
    hist_cella_t *hist = (hist_cella_t *)calloc(HIST_N, sizeof(hist_cella_t));
    histogrammum_aedifica(hist, rgb, n_pix);

    capsa_t *capsae = (capsa_t *)calloc((size_t)n_colorum, sizeof(capsa_t));
    int n_capsarum  = 1;

    capsae[0].r_min = capsae[0].g_min = capsae[0].b_min = 255;
    capsae[0].r_max = capsae[0].g_max = capsae[0].b_max = 0;

    for (int ri = 0; ri < HIST_DIM; ri++) {
        for (int gi = 0; gi < HIST_DIM; gi++) {
            for (int bi = 0; bi < HIST_DIM; bi++) {
                int hi = ri * HIST_DIM * HIST_DIM + gi * HIST_DIM + bi;
                if (hist[hi].numerus == 0)
                    continue;
                int r = ri << HIST_SHIFT;
                int g = gi << HIST_SHIFT;
                int b = bi << HIST_SHIFT;
                if (r < capsae[0].r_min)
                    capsae[0].r_min = r;
                if ((r + (1 << HIST_SHIFT) - 1) > capsae[0].r_max)
                    capsae[0].r_max = r + (1 << HIST_SHIFT) - 1;
                if (g < capsae[0].g_min)
                    capsae[0].g_min = g;
                if ((g + (1 << HIST_SHIFT) - 1) > capsae[0].g_max)
                    capsae[0].g_max = g + (1 << HIST_SHIFT) - 1;
                if (b < capsae[0].b_min)
                    capsae[0].b_min = b;
                if ((b + (1 << HIST_SHIFT) - 1) > capsae[0].b_max)
                    capsae[0].b_max = b + (1 << HIST_SHIFT) - 1;
                capsae[0].r_sum += hist[hi].r_sum;
                capsae[0].g_sum += hist[hi].g_sum;
                capsae[0].b_sum += hist[hi].b_sum;
            }
        }
    }
    capsae[0].numerus = n_pix;

    while (n_capsarum < n_colorum) {
        int optima  = -1;
        int max_amp = 0;
        for (int i = 0; i < n_capsarum; i++) {
            if (capsae[i].numerus < 2)
                continue;
            int amp = capsa_magnitudo(&capsae[i]);
            if (amp > max_amp) {
                max_amp = amp;
                optima  = i;
            }
        }
        if (optima < 0)
            break;

        int dim = capsa_amplitudo(&capsae[optima]);
        int medio;
        if (dim == 0)
            medio = (capsae[optima].r_min + capsae[optima].r_max) / 2;
        else if (dim == 1)
            medio = (capsae[optima].g_min + capsae[optima].g_max) / 2;
        else
            medio = (capsae[optima].b_min + capsae[optima].b_max) / 2;

        capsa_t lo     = {255, 0, 255, 0, 255, 0, 0, 0, 0, 0};
        capsa_t hi_box = {255, 0, 255, 0, 255, 0, 0, 0, 0, 0};

        int ri_min = capsae[optima].r_min >> HIST_SHIFT;
        int ri_max = capsae[optima].r_max >> HIST_SHIFT;
        int gi_min = capsae[optima].g_min >> HIST_SHIFT;
        int gi_max = capsae[optima].g_max >> HIST_SHIFT;
        int bi_min = capsae[optima].b_min >> HIST_SHIFT;
        int bi_max = capsae[optima].b_max >> HIST_SHIFT;
        if (ri_max >= HIST_DIM) ri_max = HIST_DIM - 1;
        if (gi_max >= HIST_DIM) gi_max = HIST_DIM - 1;
        if (bi_max >= HIST_DIM) bi_max = HIST_DIM - 1;

        for (int ri = ri_min; ri <= ri_max; ri++) {
            for (int gi = gi_min; gi <= gi_max; gi++) {
                for (int bi = bi_min; bi <= bi_max; bi++) {
                    int idx = ri * HIST_DIM * HIST_DIM + gi * HIST_DIM + bi;
                    if (hist[idx].numerus == 0)
                        continue;

                    int r = ri << HIST_SHIFT;
                    int g = gi << HIST_SHIFT;
                    int b = bi << HIST_SHIFT;
                    int v = (dim == 0) ? r : (dim == 1) ? g : b;

                    capsa_t *dest = (v <= medio) ? &lo : &hi_box;
                    int r_lo = ri << HIST_SHIFT;
                    int r_hi = r_lo + (1 << HIST_SHIFT) - 1;
                    int g_lo = gi << HIST_SHIFT;
                    int g_hi = g_lo + (1 << HIST_SHIFT) - 1;
                    int b_lo = bi << HIST_SHIFT;
                    int b_hi = b_lo + (1 << HIST_SHIFT) - 1;
                    if (r_lo < dest->r_min) dest->r_min = r_lo;
                    if (r_hi > dest->r_max) dest->r_max = r_hi;
                    if (g_lo < dest->g_min) dest->g_min = g_lo;
                    if (g_hi > dest->g_max) dest->g_max = g_hi;
                    if (b_lo < dest->b_min) dest->b_min = b_lo;
                    if (b_hi > dest->b_max) dest->b_max = b_hi;
                    dest->r_sum += hist[idx].r_sum;
                    dest->g_sum += hist[idx].g_sum;
                    dest->b_sum += hist[idx].b_sum;
                    dest->numerus += hist[idx].numerus;
                }
            }
        }

        if (lo.numerus == 0 || hi_box.numerus == 0) {
            capsae[optima].r_min = capsae[optima].r_max;
            capsae[optima].g_min = capsae[optima].g_max;
            capsae[optima].b_min = capsae[optima].b_max;
            continue;
        }

        capsae[optima]     = lo;
        capsae[n_capsarum] = hi_box;
        n_capsarum++;
    }

    for (int i = 0; i < n_colorum; i++) {
        if (i < n_capsarum && capsae[i].numerus > 0) {
            paleta[i][0] = (uint8_t)(capsae[i].r_sum / capsae[i].numerus);
            paleta[i][1] = (uint8_t)(capsae[i].g_sum / capsae[i].numerus);
            paleta[i][2] = (uint8_t)(capsae[i].b_sum / capsae[i].numerus);
        } else {
            paleta[i][0] = paleta[i][1] = paleta[i][2] = 0;
        }
    }

    free(capsae);
    free(hist);
}

/* ================================================================
 * octarboris (octree)
 *
 * arborem octonariam aedificat ex histogrammo colorum.
 * folia reducit per fusionem nodorum cum minimis pixelis,
 * ab infimo sursum procedens.
 * ================================================================ */

#define OCTA_PROF          5      /* profunditas = HIST_BITS */
#define OCTA_STAGNUM_MAG   38000  /* 1+8+64+512+4096+32768 = 37449 */

typedef struct octa_nodus {
    int64_t r_sum, g_sum, b_sum;
    int numerus;
    int n_filiorum;
    struct octa_nodus *filii[8];
} octa_nodus_t;

typedef struct {
    octa_nodus_t *stagnum;
    int stag_pos;
    int n_foliorum;
} octarboris_t;

static octa_nodus_t *octa_novum(octarboris_t *arb)
{
    octa_nodus_t *n = &arb->stagnum[arb->stag_pos++];
    memset(n, 0, sizeof(*n));
    return n;
}

static void octa_insere(
    octarboris_t *arb, int ri, int gi, int bi,
    int64_t rs, int64_t gs, int64_t bs, int num
) {
    octa_nodus_t *nod = &arb->stagnum[0];
    nod->r_sum += rs;
    nod->g_sum += gs;
    nod->b_sum += bs;
    nod->numerus += num;

    for (int d = 0; d < OCTA_PROF; d++) {
        int bit = (OCTA_PROF - 1) - d;
        int idx = ((ri >> bit) & 1) << 2 |
                  ((gi >> bit) & 1) << 1 |
                  ((bi >> bit) & 1);
        if (!nod->filii[idx]) {
            nod->filii[idx] = octa_novum(arb);
            nod->n_filiorum++;
        }
        nod = nod->filii[idx];
        nod->r_sum += rs;
        nod->g_sum += gs;
        nod->b_sum += bs;
        nod->numerus += num;
    }
    arb->n_foliorum++;
}

static int octa_folia_in(octa_nodus_t *nod)
{
    if (nod->n_filiorum == 0)
        return 1;
    int n = 0;
    for (int i = 0; i < 8; i++)
        if (nod->filii[i])
            n += octa_folia_in(nod->filii[i]);
    return n;
}

static void octa_funde(octarboris_t *arb, octa_nodus_t *nod)
{
    int folia = octa_folia_in(nod);
    for (int i = 0; i < 8; i++)
        nod->filii[i] = NULL;
    nod->n_filiorum = 0;
    arb->n_foliorum -= (folia - 1);
}

static void octa_collige_internos(
    octa_nodus_t *nod, int prof, int prof_finis,
    octa_nodus_t **lista, int *n
) {
    if (prof == prof_finis) {
        if (nod->n_filiorum > 0)
            lista[(*n)++] = nod;
        return;
    }
    for (int i = 0; i < 8; i++)
        if (nod->filii[i])
            octa_collige_internos(
                nod->filii[i], prof + 1, prof_finis, lista, n
            );
}

static int octa_comp_numerus(const void *a, const void *b)
{
    const octa_nodus_t *na = *(const octa_nodus_t *const *)a;
    const octa_nodus_t *nb = *(const octa_nodus_t *const *)b;
    return (na->numerus > nb->numerus) - (na->numerus < nb->numerus);
}

static int octa_collige_paletam(
    octa_nodus_t *nod, uint8_t paleta[][3], int pos, int max_n
) {
    if (pos >= max_n)
        return pos;
    if (nod->n_filiorum == 0) {
        if (nod->numerus > 0) {
            paleta[pos][0] = (uint8_t)(nod->r_sum / nod->numerus);
            paleta[pos][1] = (uint8_t)(nod->g_sum / nod->numerus);
            paleta[pos][2] = (uint8_t)(nod->b_sum / nod->numerus);
            return pos + 1;
        }
        return pos;
    }
    for (int i = 0; i < 8; i++)
        if (nod->filii[i])
            pos = octa_collige_paletam(nod->filii[i], paleta, pos, max_n);
    return pos;
}

void paletam_genera_octarboris(
    const uint8_t *rgb, int n_pix,
    uint8_t paleta[][3], int n_colorum
) {
    hist_cella_t *hist = (hist_cella_t *)calloc(HIST_N, sizeof(hist_cella_t));
    histogrammum_aedifica(hist, rgb, n_pix);

    octarboris_t arb;
    arb.stagnum = (octa_nodus_t *)calloc(
        OCTA_STAGNUM_MAG, sizeof(octa_nodus_t)
    );
    arb.stag_pos   = 1;
    arb.n_foliorum = 0;

    for (int ri = 0; ri < HIST_DIM; ri++)
        for (int gi = 0; gi < HIST_DIM; gi++)
            for (int bi = 0; bi < HIST_DIM; bi++) {
                int hi = ri * HIST_DIM * HIST_DIM + gi * HIST_DIM + bi;
                if (hist[hi].numerus > 0)
                    octa_insere(&arb, ri, gi, bi,
                                hist[hi].r_sum, hist[hi].g_sum,
                                hist[hi].b_sum, hist[hi].numerus);
            }

    octa_nodus_t **lista = (octa_nodus_t **)malloc(
        (size_t)arb.stag_pos * sizeof(octa_nodus_t *)
    );
    for (int d = OCTA_PROF - 1; d >= 0 && arb.n_foliorum > n_colorum; d--) {
        int n_lista = 0;
        octa_collige_internos(&arb.stagnum[0], 0, d, lista, &n_lista);
        if (n_lista == 0)
            continue;
        qsort(lista, (size_t)n_lista, sizeof(octa_nodus_t *),
              octa_comp_numerus);
        for (int i = 0; i < n_lista && arb.n_foliorum > n_colorum; i++)
            if (lista[i]->n_filiorum > 0)
                octa_funde(&arb, lista[i]);
    }
    free(lista);

    int pos = octa_collige_paletam(&arb.stagnum[0], paleta, 0, n_colorum);
    for (int i = pos; i < n_colorum; i++)
        paleta[i][0] = paleta[i][1] = paleta[i][2] = 0;

    free(arb.stagnum);
    free(hist);
}

/* ================================================================
 * k-media (k-means)
 *
 * semen ex median-cut accipit, deinde centroides iterative
 * optimat per histogrammum. convergentiam velocem praebet.
 * ================================================================ */

#define KMEDIA_ITER  15

void paletam_genera_kmedia(
    const uint8_t *rgb, int n_pix,
    uint8_t paleta[][3], int n_colorum
) {
    /* semen ex median-cut */
    paletam_genera(rgb, n_pix, paleta, n_colorum);

    hist_cella_t *hist = (hist_cella_t *)calloc(HIST_N, sizeof(hist_cella_t));
    histogrammum_aedifica(hist, rgb, n_pix);

    int64_t *summa_r = (int64_t *)calloc((size_t)n_colorum, sizeof(int64_t));
    int64_t *summa_g = (int64_t *)calloc((size_t)n_colorum, sizeof(int64_t));
    int64_t *summa_b = (int64_t *)calloc((size_t)n_colorum, sizeof(int64_t));
    int *summa_n     = (int *)calloc((size_t)n_colorum, sizeof(int));

    for (int iter = 0; iter < KMEDIA_ITER; iter++) {
        memset(summa_r, 0, (size_t)n_colorum * sizeof(int64_t));
        memset(summa_g, 0, (size_t)n_colorum * sizeof(int64_t));
        memset(summa_b, 0, (size_t)n_colorum * sizeof(int64_t));
        memset(summa_n, 0, (size_t)n_colorum * sizeof(int));

        for (int hi = 0; hi < HIST_N; hi++) {
            if (hist[hi].numerus == 0)
                continue;
            int cr = (int)(hist[hi].r_sum / hist[hi].numerus);
            int cg = (int)(hist[hi].g_sum / hist[hi].numerus);
            int cb = (int)(hist[hi].b_sum / hist[hi].numerus);

            int opt = 0, opt_dist = 0x7FFFFFFF;
            for (int c = 0; c < n_colorum; c++) {
                int dr = cr - paleta[c][0];
                int dg = cg - paleta[c][1];
                int db = cb - paleta[c][2];
                int dist = dr * dr + dg * dg + db * db;
                if (dist < opt_dist) {
                    opt_dist = dist;
                    opt      = c;
                }
            }

            summa_r[opt] += hist[hi].r_sum;
            summa_g[opt] += hist[hi].g_sum;
            summa_b[opt] += hist[hi].b_sum;
            summa_n[opt] += hist[hi].numerus;
        }

        int mutata = 0;
        for (int c = 0; c < n_colorum; c++) {
            if (summa_n[c] > 0) {
                uint8_t nr = (uint8_t)(summa_r[c] / summa_n[c]);
                uint8_t ng = (uint8_t)(summa_g[c] / summa_n[c]);
                uint8_t nb = (uint8_t)(summa_b[c] / summa_n[c]);
                if (nr != paleta[c][0] || ng != paleta[c][1] ||
                    nb != paleta[c][2]) {
                    paleta[c][0] = nr;
                    paleta[c][1] = ng;
                    paleta[c][2] = nb;
                    mutata = 1;
                }
            }
        }
        if (!mutata)
            break;
    }

    free(summa_r);
    free(summa_g);
    free(summa_b);
    free(summa_n);
    free(hist);
}
