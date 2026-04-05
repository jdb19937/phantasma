/*
 * quantisatio.h — algorithmi quantisationis colorum
 *
 * Tres modos praebet:
 *   median-cut, octarboris (octree), k-media (k-means).
 */

#ifndef QUANTISATIO_H
#define QUANTISATIO_H

#include <stdint.h>

void paletam_genera(
    const uint8_t *rgb, int n_pix,
    uint8_t paleta[][3], int n_colorum
);

void paletam_genera_octarboris(
    const uint8_t *rgb, int n_pix,
    uint8_t paleta[][3], int n_colorum
);

void paletam_genera_kmedia(
    const uint8_t *rgb, int n_pix,
    uint8_t paleta[][3], int n_colorum
);

#endif /* QUANTISATIO_H */
