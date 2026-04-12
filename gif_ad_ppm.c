/*
 * gif_ad_ppm.c — GIF ad PPM conversor
 *
 * Usus:
 *   gif_ad_ppm imago.gif
 *
 * Si imago unam tabulam habet, scribit imago.ppm.
 * Si plures tabulas habet, scribit imago_000.ppm, imago_001.ppm, ...
 */

#include "phantasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ppm_scribe(const char *via, const uint32_t *pixels, int lat, int alt)
{
    FILE *f = fopen(via, "wb");
    if (!f) {
        fprintf(stderr, "erratum: '%s' aperire non possum\n", via);
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", lat, alt);

    int n = lat * alt;
    for (int i = 0; i < n; i++) {
        uint32_t p = pixels[i];
        unsigned char rgb[3];
        rgb[0] = (p >> 16) & 0xFF;
        rgb[1] = (p >>  8) & 0xFF;
        rgb[2] =  p        & 0xFF;
        fwrite(rgb, 1, 3, f);
    }

    fclose(f);
}

/* "imago.gif" -> "imago", extensionem removet */
static void basim_excerpe(const char *via, char *basis, size_t mag)
{
    const char *ultimus_sep = strrchr(via, '/');
    const char *nomen       = ultimus_sep ? ultimus_sep + 1 : via;
    const char *punctum     = strrchr(nomen, '.');

    size_t lon = punctum ? (size_t)(punctum - nomen) : strlen(nomen);
    if (lon >= mag)
        lon = mag - 1;
    memcpy(basis, nomen, lon);
    basis[lon] = '\0';
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usus: %s <imago.gif>\n", argv[0]);
        return 1;
    }

    const char *via     = argv[1];
    pfr_gif_lector_t *l = pfr_gif_lege_initia(via);
    if (!l) {
        fprintf(stderr, "erratum: '%s' legere non possum\n", via);
        return 1;
    }

    int lat, alt;
    pfr_gif_lege_dimensiones(l, &lat, &alt);

    uint32_t *pixels = malloc((size_t)lat * alt * sizeof(uint32_t));
    if (!pixels) {
        fprintf(stderr, "erratum: memoria non sufficit\n");
        pfr_gif_lege_fini(l);
        return 1;
    }

    /* primum omnes tabulas lege ut numerum sciamus */
    int numerus        = 0;
    uint32_t **tabulae = NULL;

    while (pfr_gif_lege_tabulam(l, pixels) == 0) {
        uint32_t *copia = malloc((size_t)lat * alt * sizeof(uint32_t));
        if (!copia) {
            fprintf(stderr, "erratum: memoria non sufficit\n");
            break;
        }
        memcpy(copia, pixels, (size_t)lat * alt * sizeof(uint32_t));

        uint32_t **nova = realloc(tabulae, (size_t)(numerus + 1) * sizeof(uint32_t *));
        if (!nova) {
            free(copia);
            break;
        }
        tabulae = nova;
        tabulae[numerus++] = copia;
    }

    pfr_gif_lege_fini(l);
    free(pixels);

    if (numerus == 0) {
        fprintf(stderr, "erratum: nullas tabulas legere potui\n");
        free(tabulae);
        return 1;
    }

    char basis[256];
    basim_excerpe(via, basis, sizeof(basis));

    char via_ppm[512];

    if (numerus == 1) {
        snprintf(via_ppm, sizeof(via_ppm), "%s.ppm", basis);
        ppm_scribe(via_ppm, tabulae[0], lat, alt);
        fprintf(stderr, "%s (%dx%d)\n", via_ppm, lat, alt);
    } else {
        for (int i = 0; i < numerus; i++) {
            snprintf(via_ppm, sizeof(via_ppm), "%s_%03d.ppm", basis, i);
            ppm_scribe(via_ppm, tabulae[i], lat, alt);
            fprintf(stderr, "%s (%dx%d)\n", via_ppm, lat, alt);
        }
    }

    for (int i = 0; i < numerus; i++)
        free(tabulae[i]);
    free(tabulae);

    return 0;
}
