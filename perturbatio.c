/*
 * perturbatio.c — algorithmi perturbationis (dithering)
 * ======================================================
 *
 * Pixelos RGB in indices paletae convertit cum perturbatione
 * optionali ad artefacta quantisationis reducenda.
 *
 * Modi:
 *   1. Bayer 8x8  — perturbatio ordinata, velocissima
 *   2. Floyd-Steinberg — error diffusion, qualitas optima
 *   3. Nullum     — sine perturbatione, proximus color directe
 */

#include "perturbatio.h"

#include <stdlib.h>
#include <string.h>

/* ================================================================
 * matrica Bayer 8x8
 * ================================================================ */

static const int bayer_8x8[8][8] = {
    {  0, 32,  8, 40,  2, 34, 10, 42 },
    { 48, 16, 56, 24, 50, 18, 58, 26 },
    { 12, 44,  4, 36, 14, 46,  6, 38 },
    { 60, 28, 52, 20, 62, 30, 54, 22 },
    {  3, 35, 11, 43,  1, 33,  9, 41 },
    { 51, 19, 59, 27, 49, 17, 57, 25 },
    { 15, 47,  7, 39, 13, 45,  5, 37 },
    { 63, 31, 55, 23, 61, 29, 53, 21 }
};

/* ================================================================
 * indicem proximum in paleta invenire
 * ================================================================ */

static int indicem_proximum(
    const uint8_t paleta[][3], int n,
    int r, int g, int b
) {
    int optimus  = 0;
    int min_dist = 0x7FFFFFFF;
    for (int i = 0; i < n; i++) {
        int dr   = r - paleta[i][0];
        int dg   = g - paleta[i][1];
        int db   = b - paleta[i][2];
        int dist = dr * dr + dg * dg + db * db;
        if (dist < min_dist) {
            min_dist = dist;
            optimus  = i;
        }
    }
    return optimus;
}

/* ================================================================
 * tabula quaesitionis celeris — 32x32x32 (32KB)
 *
 * colores RGB ad 5 bits truncat, indicem paletae recondit.
 * primam quaesitionem lentam facit, repetitas velocissimas.
 * ================================================================ */

#define CELER_BITS  5
#define CELER_MAG   (1 << CELER_BITS)   /* 32 */
#define CELER_SHIFT (8 - CELER_BITS)    /* 3 */

typedef struct {
    uint8_t indices[CELER_MAG][CELER_MAG][CELER_MAG];
    uint8_t repleta[CELER_MAG][CELER_MAG][CELER_MAG];
} celer_tabula_t;

static void celer_initia(celer_tabula_t *ct)
{
    memset(ct->repleta, 0, sizeof(ct->repleta));
}

static int celer_quaere(
    celer_tabula_t *ct, const uint8_t paleta[][3],
    int n, int r, int g, int b
) {
    int ri = r >> CELER_SHIFT;
    int gi = g >> CELER_SHIFT;
    int bi = b >> CELER_SHIFT;
    if (ct->repleta[ri][gi][bi])
        return ct->indices[ri][gi][bi];

    int idx = indicem_proximum(paleta, n, r, g, b);
    ct->indices[ri][gi][bi] = (uint8_t)idx;
    ct->repleta[ri][gi][bi] = 1;
    return idx;
}

/* ================================================================
 * Bayer 8x8 — perturbatio ordinata
 * ================================================================ */

void indices_bayer(
    const uint8_t *rgb, int lat, int alt,
    const uint8_t paleta[][3], int n_pal,
    uint8_t *indices
) {
    celer_tabula_t *ct = (celer_tabula_t *)malloc(sizeof(celer_tabula_t));
    celer_initia(ct);

    for (int y = 0; y < alt; y++) {
        for (int x = 0; x < lat; x++) {
            int idx = (y * lat + x) * 3;
            double limen  = (bayer_8x8[y & 7][x & 7] / 64.0) - 0.5;
            double campus = 24.0;
            int r  = (int)(rgb[idx + 0] + limen * campus);
            int gv = (int)(rgb[idx + 1] + limen * campus);
            int b  = (int)(rgb[idx + 2] + limen * campus);
            if (r < 0)   r = 0;   if (r > 255)   r = 255;
            if (gv < 0)  gv = 0;  if (gv > 255)  gv = 255;
            if (b < 0)   b = 0;   if (b > 255)   b = 255;
            indices[y * lat + x] = (uint8_t)celer_quaere(
                ct, paleta, n_pal, r, gv, b
            );
        }
    }
    free(ct);
}

/* ================================================================
 * Floyd-Steinberg — error diffusion
 *
 * errorem quantisationis ad vicinos propagat:
 *   (x+1, y)   += error * 7/16
 *   (x-1, y+1) += error * 3/16
 *   (x,   y+1) += error * 5/16
 *   (x+1, y+1) += error * 1/16
 * ================================================================ */

void indices_floyd(
    const uint8_t *rgb, int lat, int alt,
    const uint8_t paleta[][3], int n_pal,
    uint8_t *indices
) {
    int n = lat * alt;
    int16_t *alveus = (int16_t *)malloc((size_t)n * 3 * sizeof(int16_t));
    for (int i = 0; i < n * 3; i++)
        alveus[i] = rgb[i];

    for (int y = 0; y < alt; y++) {
        for (int x = 0; x < lat; x++) {
            int p  = (y * lat + x) * 3;
            int r  = alveus[p + 0] < 0 ? 0 :
                    (alveus[p + 0] > 255 ? 255 : alveus[p + 0]);
            int gv = alveus[p + 1] < 0 ? 0 :
                    (alveus[p + 1] > 255 ? 255 : alveus[p + 1]);
            int b  = alveus[p + 2] < 0 ? 0 :
                    (alveus[p + 2] > 255 ? 255 : alveus[p + 2]);

            int ci = indicem_proximum(paleta, n_pal, r, gv, b);
            indices[y * lat + x] = (uint8_t)ci;

            int er = r - paleta[ci][0];
            int eg = gv - paleta[ci][1];
            int eb = b - paleta[ci][2];

            /* propaga errorem ad vicinos */
            if (x + 1 < lat) {
                int j = p + 3;
                alveus[j + 0] += (int16_t)(er * 7 / 16);
                alveus[j + 1] += (int16_t)(eg * 7 / 16);
                alveus[j + 2] += (int16_t)(eb * 7 / 16);
            }
            if (y + 1 < alt) {
                if (x > 0) {
                    int j = ((y + 1) * lat + (x - 1)) * 3;
                    alveus[j + 0] += (int16_t)(er * 3 / 16);
                    alveus[j + 1] += (int16_t)(eg * 3 / 16);
                    alveus[j + 2] += (int16_t)(eb * 3 / 16);
                }
                {
                    int j = ((y + 1) * lat + x) * 3;
                    alveus[j + 0] += (int16_t)(er * 5 / 16);
                    alveus[j + 1] += (int16_t)(eg * 5 / 16);
                    alveus[j + 2] += (int16_t)(eb * 5 / 16);
                }
                if (x + 1 < lat) {
                    int j = ((y + 1) * lat + (x + 1)) * 3;
                    alveus[j + 0] += (int16_t)(er / 16);
                    alveus[j + 1] += (int16_t)(eg / 16);
                    alveus[j + 2] += (int16_t)(eb / 16);
                }
            }
        }
    }
    free(alveus);
}

/* ================================================================
 * nullum — sine perturbatione
 * ================================================================ */

void indices_nullum(
    const uint8_t *rgb, int lat, int alt,
    const uint8_t paleta[][3], int n_pal,
    uint8_t *indices
) {
    celer_tabula_t *ct = (celer_tabula_t *)malloc(sizeof(celer_tabula_t));
    celer_initia(ct);

    for (int y = 0; y < alt; y++) {
        for (int x = 0; x < lat; x++) {
            int idx = (y * lat + x) * 3;
            indices[y * lat + x] = (uint8_t)celer_quaere(
                ct, paleta, n_pal,
                rgb[idx + 0], rgb[idx + 1], rgb[idx + 2]
            );
        }
    }
    free(ct);
}
