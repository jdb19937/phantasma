/*
 * perturbatio.h — algorithmi perturbationis (dithering)
 *
 * Tres modos praebet:
 *   Bayer 8x8 ordinatus, Floyd-Steinberg error diffusion, nullum.
 */

#ifndef PERTURBATIO_H
#define PERTURBATIO_H

#include <stdint.h>

void indices_bayer(
    const uint8_t *rgb, int lat, int alt,
    const uint8_t paleta[][3], int n_pal,
    uint8_t *indices
);

void indices_floyd(
    const uint8_t *rgb, int lat, int alt,
    const uint8_t paleta[][3], int n_pal,
    uint8_t *indices
);

void indices_nullum(
    const uint8_t *rgb, int lat, int alt,
    const uint8_t paleta[][3], int n_pal,
    uint8_t *indices
);

#endif /* PERTURBATIO_H */
