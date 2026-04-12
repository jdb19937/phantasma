/*
 * proba_gif.c — probationes unitariae inscriptoris GIF
 *
 * Compilatio:
 *   cc -Wall -Wextra -pedantic -std=c99 -O2 -c pfr_gif.c -o pfr_gif.o
 *   cc -Wall -Wextra -pedantic -std=c99 -O2 proba_gif.c pfr_gif.o -lm -o proba_gif
 *
 * Usus:
 *   ./proba_gif
 */

#include "phantasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================
 * instrumenta probationis
 * ================================================================ */

static int probationes_totae  = 0;
static int probationes_rectae = 0;
static int probationes_falsae = 0;

#define PROBA(nomen, corpus) do {                                   \
    probationes_totae++;                                            \
    const char *_err = (corpus);                                    \
    if (_err) {                                                     \
        probationes_falsae++;                                       \
        fprintf(stderr, "  FALSUM: %s — %s\n", (nomen), _err);     \
    } else {                                                        \
        probationes_rectae++;                                       \
        fprintf(stderr, "  rectum: %s\n", (nomen));                 \
    }                                                               \
} while (0)

static char via_tmp[256];
static int via_index = 0;

static const char *viam_novam(void)
{
    snprintf(via_tmp, sizeof(via_tmp), "/tmp/proba_gif_%d.gif", via_index++);
    return via_tmp;
}

static void plicam_dele(const char *via)
{
    remove(via);
}

static uint32_t *pixels_crea(int lat, int alt, uint32_t color)
{
    size_t n    = (size_t)lat * alt;
    uint32_t *p = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!p)
        return NULL;
    for (size_t i = 0; i < n; i++)
        p[i] = color;
    return p;
}

static long plica_magnitudo(const char *via)
{
    FILE *f = fopen(via, "rb");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    long mag = ftell(f);
    fclose(f);
    return mag;
}

/* verifica caput GIF89a */
static int est_gif89a(const char *via)
{
    FILE *f = fopen(via, "rb");
    if (!f)
        return 0;
    char caput[6];
    if (fread(caput, 1, 6, f) != 6) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return memcmp(caput, "GIF89a", 6) == 0;
}

/* ================================================================
 * probationes
 * ================================================================ */

/* --- parametra nulla --- */

static const char *proba_initia_via_nulla(void)
{
    pfr_gif_t *g = pfr_gif_initia(NULL, 64, 64, 3, 1);
    if (g) {
        pfr_gif_fini(g);
        return "non debet initiari cum via nulla";
    }
    return NULL;
}

static const char *proba_adde_nulla(void)
{
    if (pfr_gif_tabulam_adde(NULL, NULL) != -1)
        return "debet -1 reddere cum inscriptore nullo";
    return NULL;
}

static const char *proba_fini_nulla(void)
{
    pfr_gif_fini(NULL);
    return NULL;
}

static const char *proba_pixels_nullus(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 32, 32, 3, 1);
    if (!g)
        return "initia falsum";

    int res = pfr_gif_tabulam_adde(g, NULL);
    pfr_gif_fini(g);
    plicam_dele(via);

    if (res != -1)
        return "debet -1 reddere cum pixelibus nullis";
    return NULL;
}

/* --- vita basica --- */

static const char *proba_vita_simplex(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 64, 64, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(64, 64, 0xFFFF0000);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);

    if (!est_gif89a(via)) {
        plicam_dele(via);
        return "non est GIF89a";
    }

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

static const char *proba_plures_tabulae(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 128, 96, 5, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(128, 96, 0xFF00FF00);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    for (int i = 0; i < 10; i++) {
        /* muta colorem */
        for (int j = 0; j < 128 * 96; j++)
            pix[j] = 0xFF000000 | (uint32_t)((i * 25) << 16) |
                (uint32_t)((i * 15) << 8) | (uint32_t)(j & 0xFF);

        int res = pfr_gif_tabulam_adde(g, pix);
        if (res != 0) {
            free(pix);
            pfr_gif_fini(g);
            plicam_dele(via);
            return "adde falsum";
        }
    }

    free(pix);
    pfr_gif_fini(g);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

static const char *proba_sine_tabulis(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 32, 32, 3, 1);
    if (!g)
        return "initia falsum";

    pfr_gif_fini(g);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag < 0)
        return "plica non scripta";
    return NULL;
}

/* --- scala --- */

static const char *proba_scala_2(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 128, 128, 3, 2);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(128, 128, 0xFFAA5533);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);
    if (!est_gif89a(via)) {
        plicam_dele(via);
        return "non est GIF89a";
    }

    plicam_dele(via);
    return NULL;
}

static const char *proba_scala_4(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 256, 256, 3, 4);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(256, 256, 0xFF112233);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);
    plicam_dele(via);
    return NULL;
}

/* --- dimensiones --- */

static const char *proba_dimensio_1x1(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 1, 1, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t pix = 0xFFFFFFFF;
    int res      = pfr_gif_tabulam_adde(g, &pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);
    plicam_dele(via);
    return NULL;
}

static const char *proba_dimensio_3x3(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 3, 3, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t pix[9];
    for (int i = 0; i < 9; i++)
        pix[i] = 0xFF000000 | (uint32_t)(i * 28) << 16;

    int res = pfr_gif_tabulam_adde(g, pix);
    pfr_gif_fini(g);
    plicam_dele(via);

    if (res != 0)
        return "adde falsum cum 3x3";
    return NULL;
}

/* --- 768x768 scala 2 (sicut excrashy.c orbita) --- */

static const char *proba_dimensio_768_scala2(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 768, 768, 3, 2);
    if (!g)
        return "initia falsum";

    uint32_t *pix = (uint32_t *)malloc(768 * 768 * sizeof(uint32_t));
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    /* gradiens — colores diversi */
    for (int y = 0; y < 768; y++)
        for (int x = 0; x < 768; x++)
            pix[y * 768 + x] = 0xFF000000 |
                ((uint32_t)(x & 0xFF) << 16) |
                ((uint32_t)(y & 0xFF) << 8) |
                (uint32_t)((x + y) & 0xFF);

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- GIF trailer (0x3B) --- */

static const char *proba_trailer(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 32, 32, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(32, 32, 0xFF804020);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }
    pfr_gif_tabulam_adde(g, pix);
    free(pix);
    pfr_gif_fini(g);

    FILE *f = fopen(via, "rb");
    plicam_dele(via);
    if (!f)
        return "plica non aperitur";

    fseek(f, -1, SEEK_END);
    int ultimus = fgetc(f);
    fclose(f);

    if (ultimus != 0x3B)
        return "GIF trailer (0x3B) non inventus ad finem";
    return NULL;
}

/* --- colores diversi (median-cut) --- */

static const char *proba_colores_diversi(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 64, 64, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = (uint32_t *)malloc(64 * 64 * sizeof(uint32_t));
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    /* imago cum multis coloribus diversis */
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
            pix[y * 64 + x] = 0xFF000000 |
                ((uint32_t)(x * 4) << 16) |
                ((uint32_t)(y * 4) << 8) |
                (uint32_t)((x + y) * 2);

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- celeritas cum tabula quaesitionis --- */

static const char *proba_celeritas(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 384, 384, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = (uint32_t *)malloc(384 * 384 * sizeof(uint32_t));
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    for (int y = 0; y < 384; y++)
        for (int x = 0; x < 384; x++)
            pix[y * 384 + x] = 0xFF000000 |
                ((uint32_t)(x & 0xFF) << 16) |
                ((uint32_t)(y & 0xFF) << 8) |
                (uint32_t)((x * y) & 0xFF);

    clock_t ante = clock();

    for (int i = 0; i < 5; i++) {
        int res = pfr_gif_tabulam_adde(g, pix);
        if (res != 0) {
            free(pix);
            pfr_gif_fini(g);
            plicam_dele(via);
            return "adde falsum";
        }
    }

    clock_t post    = clock();
    double tempus_s = (double)(post - ante) / CLOCKS_PER_SEC;

    free(pix);
    pfr_gif_fini(g);
    plicam_dele(via);

    fprintf(stderr, " (%.3f s pro 5 tabulis 384x384)", tempus_s);

    /* debet sub 2 secundis esse (cum tabula celeri) */
    if (tempus_s > 2.0)
        return "nimis lentum — tabula quaesitionis non operatur?";
    return NULL;
}

/* --- mora diversa --- */

static const char *proba_mora_diversa(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 32, 32, 100, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(32, 32, 0xFF0000FF);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    pfr_gif_tabulam_adde(g, pix);
    pfr_gif_tabulam_adde(g, pix);
    free(pix);
    pfr_gif_fini(g);

    if (!est_gif89a(via)) {
        plicam_dele(via);
        return "non est GIF89a";
    }

    plicam_dele(via);
    return NULL;
}

/* --- imago unicolor (omnes pixels idem) --- */

static const char *proba_unicolor(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 64, 64, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(64, 64, 0xFF808080);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);
    plicam_dele(via);
    return NULL;
}

/* --- imago nigra --- */

static const char *proba_nigra(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 64, 64, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(64, 64, 0xFF000000);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    pfr_gif_fini(g);
    plicam_dele(via);

    if (res != 0)
        return "adde falsum cum imagine nigra";
    return NULL;
}

/* ================================================================
 * probationes modorum — quantisatio et dithering
 * ================================================================ */

static uint32_t *pixels_gradiens(int lat, int alt)
{
    uint32_t *pix = (uint32_t *)malloc((size_t)lat * alt * sizeof(uint32_t));
    if (!pix)
        return NULL;
    for (int y = 0; y < alt; y++)
        for (int x = 0; x < lat; x++)
            pix[y * lat + x] = 0xFF000000 |
                ((uint32_t)(x * 4) << 16) |
                ((uint32_t)(y * 4) << 8) |
                (uint32_t)((x + y) * 2);
    return pix;
}

static const char *proba_modus(int quant, int dither)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 64, 64, 3, 1);
    if (!g)
        return "initia falsum";

    pfr_gif_modum_pone(g, (pfr_quant_t)quant, (pfr_dither_t)dither);

    uint32_t *pix = pixels_gradiens(64, 64);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    if (res != 0) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_gif_fini(g);
    if (!est_gif89a(via)) {
        plicam_dele(via);
        return "non est GIF89a";
    }
    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

static const char *proba_octarboris_bayer(void)
{ return proba_modus(PFR_QUANT_OCTARBORIS, PFR_DITHER_BAYER); }

static const char *proba_octarboris_floyd(void)
{ return proba_modus(PFR_QUANT_OCTARBORIS, PFR_DITHER_FLOYD); }

static const char *proba_octarboris_nullum(void)
{ return proba_modus(PFR_QUANT_OCTARBORIS, PFR_DITHER_NULLUM); }

static const char *proba_kmedia_bayer(void)
{ return proba_modus(PFR_QUANT_KMEDIA, PFR_DITHER_BAYER); }

static const char *proba_kmedia_floyd(void)
{ return proba_modus(PFR_QUANT_KMEDIA, PFR_DITHER_FLOYD); }

static const char *proba_kmedia_nullum(void)
{ return proba_modus(PFR_QUANT_KMEDIA, PFR_DITHER_NULLUM); }

static const char *proba_mediana_floyd(void)
{ return proba_modus(PFR_QUANT_MEDIANA, PFR_DITHER_FLOYD); }

static const char *proba_mediana_nullum(void)
{ return proba_modus(PFR_QUANT_MEDIANA, PFR_DITHER_NULLUM); }

static const char *proba_omnes_modi(void)
{
    for (int q = 0; q <= 2; q++)
        for (int d = 0; d <= 2; d++) {
        const char *err = proba_modus(q, d);
        if (err)
            return err;
    }
    return NULL;
}

/* --- plures tabulae cum modis diversis --- */

static const char *proba_plures_octarboris(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 64, 64, 3, 1);
    if (!g)
        return "initia falsum";

    pfr_gif_modum_pone(g, PFR_QUANT_OCTARBORIS, PFR_DITHER_FLOYD);

    for (int i = 0; i < 5; i++) {
        uint32_t *pix = pixels_gradiens(64, 64);
        if (!pix) {
            pfr_gif_fini(g);
            plicam_dele(via);
            return "memoria";
        }
        /* muta colores per frame */
        for (int j = 0; j < 64 * 64; j++)
            pix[j] = (pix[j] + (uint32_t)(i * 0x111111)) | 0xFF000000;

        int res = pfr_gif_tabulam_adde(g, pix);
        free(pix);
        if (res != 0) {
            pfr_gif_fini(g);
            plicam_dele(via);
            return "adde falsum";
        }
    }

    pfr_gif_fini(g);
    if (!est_gif89a(via)) {
        plicam_dele(via);
        return "non est GIF89a";
    }
    plicam_dele(via);
    return NULL;
}

/* ================================================================
 * probationes pellucditatis
 * ================================================================ */

static const char *proba_pelluc_simplex(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 4, 4, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t pix[16];
    /* duo pixeli pellucidi, duo opaci */
    for (int i = 0; i < 16; i++) {
        if (i < 8)
            pix[i] = 0xFFFF0000;   /* rubrum opacum */
        else
            pix[i] = 0x00000000;   /* pellucidus */
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    pfr_gif_fini(g);

    if (res != 0) {
        plicam_dele(via);
        return "adde falsum cum pelluciditate";
    }
    if (!est_gif89a(via)) {
        plicam_dele(via);
        return "non est GIF89a";
    }

    /* verifica GCE pelluciditatem habet */
    FILE *f = fopen(via, "rb");
    plicam_dele(via);
    if (!f)
        return "plica non aperitur";

    /* quaere GCE (0x21 0xF9) */
    int inventa = 0;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == 0x21) {
            int label = fgetc(f);
            if (label == 0xF9) {
                fgetc(f);   /* mag */
                int packed = fgetc(f);
                if (packed & 0x01)
                    inventa = 1;
                break;
            }
        }
    }
    fclose(f);

    if (!inventa)
        return "GCE pelluciditas non inventa";
    return NULL;
}

static const char *proba_pelluc_omnes(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 8, 8, 3, 1);
    if (!g)
        return "initia falsum";

    /* omnes pixeli pellucidi */
    uint32_t pix[64];
    for (int i = 0; i < 64; i++)
        pix[i] = 0x00000000;

    int res = pfr_gif_tabulam_adde(g, pix);
    pfr_gif_fini(g);
    plicam_dele(via);

    if (res != 0)
        return "adde falsum cum omnibus pellucidis";
    return NULL;
}

static const char *proba_pelluc_gradiens(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 64, 64, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = (uint32_t *)malloc(64 * 64 * sizeof(uint32_t));
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    /* gradiens alpha: supra opacus, infra pellucidus */
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
        uint32_t a      = (uint32_t)(255 - y * 4);
        pix[y * 64 + x] = (a << 24) | 0x00FF8040;
    }

    int res = pfr_gif_tabulam_adde(g, pix);
    free(pix);
    pfr_gif_fini(g);
    plicam_dele(via);

    if (res != 0)
        return "adde falsum cum gradiente alpha";
    return NULL;
}

static const char *proba_sine_pelluc(void)
{
    /* imago sine pellucditate — GCE non debet pelluciditatem habere */
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 4, 4, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t pix[16];
    for (int i = 0; i < 16; i++)
        pix[i] = 0xFFFF0000;

    pfr_gif_tabulam_adde(g, pix);
    pfr_gif_fini(g);

    FILE *f = fopen(via, "rb");
    plicam_dele(via);
    if (!f)
        return "plica non aperitur";

    int pelluc_inventa = 0;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == 0x21) {
            int label = fgetc(f);
            if (label == 0xF9) {
                fgetc(f);
                int packed = fgetc(f);
                if (packed & 0x01)
                    pelluc_inventa = 1;
                break;
            }
        }
    }
    fclose(f);

    if (pelluc_inventa)
        return "pelluciditas inventa in imagine opaca";
    return NULL;
}

/* ================================================================
 * probationes lectoris (decoder)
 * ================================================================ */

static const char *proba_lector_simplex(void)
{
    /* scribe GIF, deinde lege */
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 4, 4, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t pix_fons[16];
    for (int i = 0; i < 16; i++)
        pix_fons[i] = 0xFFFF0000;

    pfr_gif_tabulam_adde(g, pix_fons);
    pfr_gif_fini(g);

    /* lege */
    pfr_gif_lector_t *l = pfr_gif_lege_initia(via);
    if (!l) {
        plicam_dele(via);
        return "lector initia falsum";
    }

    int lat, alt;
    pfr_gif_lege_dimensiones(l, &lat, &alt);
    if (lat != 4 || alt != 4) {
        pfr_gif_lege_fini(l);
        plicam_dele(via);
        return "dimensiones falsae";
    }

    uint32_t pix_lecta[16];
    int res = pfr_gif_lege_tabulam(l, pix_lecta);
    if (res != 0) {
        pfr_gif_lege_fini(l);
        plicam_dele(via);
        return "lege tabulam falsum";
    }

    /* verifica quod omnes pixeli sunt opaci et rubri */
    for (int i = 0; i < 16; i++) {
        if ((pix_lecta[i] >> 24) != 0xFF) {
            pfr_gif_lege_fini(l);
            plicam_dele(via);
            return "alpha non opacus";
        }
        /* rubrum debet esse proximus colori originali (quantisatio) */
        int r = (pix_lecta[i] >> 16) & 0xFF;
        if (r < 200) {
            pfr_gif_lege_fini(l);
            plicam_dele(via);
            return "color ruber non proximus";
        }
    }

    /* secunda tabula non debet existere */
    res = pfr_gif_lege_tabulam(l, pix_lecta);
    if (res != -1) {
        pfr_gif_lege_fini(l);
        plicam_dele(via);
        return "debet -1 reddere post ultimam tabulam";
    }

    pfr_gif_lege_fini(l);
    plicam_dele(via);
    return NULL;
}

static const char *proba_lector_pelluc(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 4, 2, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t pix_fons[8];
    /* prima linea opaca, secunda pellucida */
    for (int i = 0; i < 4; i++)
        pix_fons[i] = 0xFF00FF00;   /* viridis opacus */
    for (int i = 4; i < 8; i++)
        pix_fons[i] = 0x00000000;   /* pellucidus */

    pfr_gif_tabulam_adde(g, pix_fons);
    pfr_gif_fini(g);

    pfr_gif_lector_t *l = pfr_gif_lege_initia(via);
    if (!l) {
        plicam_dele(via);
        return "lector initia falsum";
    }

    uint32_t pix_lecta[8];
    int res = pfr_gif_lege_tabulam(l, pix_lecta);
    if (res != 0) {
        pfr_gif_lege_fini(l);
        plicam_dele(via);
        return "lege falsum";
    }

    /* verifica: prima linea opaca */
    for (int i = 0; i < 4; i++) {
        if ((pix_lecta[i] >> 24) != 0xFF) {
            pfr_gif_lege_fini(l);
            plicam_dele(via);
            return "prima linea debet esse opaca";
        }
    }

    /* verifica: secunda linea pellucida */
    for (int i = 4; i < 8; i++) {
        if ((pix_lecta[i] >> 24) != 0x00) {
            pfr_gif_lege_fini(l);
            plicam_dele(via);
            return "secunda linea debet esse pellucida";
        }
    }

    pfr_gif_lege_fini(l);
    plicam_dele(via);
    return NULL;
}

static const char *proba_lector_plures(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 8, 8, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(8, 8, 0xFF0000FF);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 64; j++)
            pix[j] = 0xFF000000 | (uint32_t)(i * 50) << 8;
        pfr_gif_tabulam_adde(g, pix);
    }
    free(pix);
    pfr_gif_fini(g);

    pfr_gif_lector_t *l = pfr_gif_lege_initia(via);
    if (!l) {
        plicam_dele(via);
        return "lector initia falsum";
    }

    uint32_t lecta[64];
    int numerus = 0;
    while (pfr_gif_lege_tabulam(l, lecta) == 0)
        numerus++;

    pfr_gif_lege_fini(l);
    plicam_dele(via);

    if (numerus != 5)
        return "numerus tabularum falsus";
    return NULL;
}

static const char *proba_lector_nulla(void)
{
    pfr_gif_lector_t *l = pfr_gif_lege_initia(NULL);
    if (l) {
        pfr_gif_lege_fini(l);
        return "non debet initiari cum via nulla";
    }
    if (pfr_gif_lege_tabulam(NULL, NULL) != -1)
        return "debet -1 reddere cum lectore nullo";
    pfr_gif_lege_fini(NULL);
    return NULL;
}

static const char *proba_lector_dimensiones(void)
{
    const char *via = viam_novam();
    pfr_gif_t *g    = pfr_gif_initia(via, 100, 50, 3, 1);
    if (!g)
        return "initia falsum";

    uint32_t *pix = pixels_crea(100, 50, 0xFF808080);
    if (!pix) {
        pfr_gif_fini(g);
        plicam_dele(via);
        return "memoria";
    }
    pfr_gif_tabulam_adde(g, pix);
    free(pix);
    pfr_gif_fini(g);

    pfr_gif_lector_t *l = pfr_gif_lege_initia(via);
    if (!l) {
        plicam_dele(via);
        return "lector initia falsum";
    }

    int lat = 0, alt = 0;
    pfr_gif_lege_dimensiones(l, &lat, &alt);
    pfr_gif_lege_fini(l);
    plicam_dele(via);

    if (lat != 100 || alt != 50)
        return "dimensiones falsae";
    return NULL;
}

/* ================================================================
 * principium
 * ================================================================ */

int main(void)
{
    fprintf(stderr, "=== Probationes inscriptoris GIF ===\n\n");

    fprintf(stderr, "— parametra nulla —\n");
    PROBA("initia via nulla",      proba_initia_via_nulla());
    PROBA("adde inscriptor nullus", proba_adde_nulla());
    PROBA("fini nullus",           proba_fini_nulla());
    PROBA("pixels nullus",         proba_pixels_nullus());

    fprintf(stderr, "\n— vita basica —\n");
    PROBA("vita simplex",          proba_vita_simplex());
    PROBA("plures tabulae",        proba_plures_tabulae());
    PROBA("sine tabulis",          proba_sine_tabulis());

    fprintf(stderr, "\n— scala —\n");
    PROBA("scala 2",               proba_scala_2());
    PROBA("scala 4",               proba_scala_4());

    fprintf(stderr, "\n— dimensiones —\n");
    PROBA("1x1",                   proba_dimensio_1x1());
    PROBA("3x3",                   proba_dimensio_3x3());
    PROBA("768x768 scala 2",       proba_dimensio_768_scala2());

    fprintf(stderr, "\n— structura GIF —\n");
    PROBA("caput GIF89a",          proba_vita_simplex());
    PROBA("trailer 0x3B",          proba_trailer());

    fprintf(stderr, "\n— colores —\n");
    PROBA("colores diversi",       proba_colores_diversi());
    PROBA("unicolor",              proba_unicolor());
    PROBA("nigra",                 proba_nigra());
    PROBA("mora diversa",          proba_mora_diversa());

    fprintf(stderr, "\n— celeritas —\n");
    PROBA("celeritas (5x384x384)", proba_celeritas());

    fprintf(stderr, "\n— modi quantisationis —\n");
    PROBA("octarboris + bayer",    proba_octarboris_bayer());
    PROBA("octarboris + floyd",    proba_octarboris_floyd());
    PROBA("octarboris + nullum",   proba_octarboris_nullum());
    PROBA("kmedia + bayer",        proba_kmedia_bayer());
    PROBA("kmedia + floyd",        proba_kmedia_floyd());
    PROBA("kmedia + nullum",       proba_kmedia_nullum());
    PROBA("mediana + floyd",       proba_mediana_floyd());
    PROBA("mediana + nullum",      proba_mediana_nullum());
    PROBA("omnes modi (3x3)",      proba_omnes_modi());
    PROBA("plures tab octarboris", proba_plures_octarboris());

    fprintf(stderr, "\n— pelluciditas —\n");
    PROBA("pelluc simplex",        proba_pelluc_simplex());
    PROBA("pelluc omnes",          proba_pelluc_omnes());
    PROBA("pelluc gradiens",       proba_pelluc_gradiens());
    PROBA("sine pelluciditate",    proba_sine_pelluc());

    fprintf(stderr, "\n— lector (decoder) —\n");
    PROBA("lector nullus",         proba_lector_nulla());
    PROBA("lector simplex",        proba_lector_simplex());
    PROBA("lector pellucidus",     proba_lector_pelluc());
    PROBA("lector plures tabulae", proba_lector_plures());
    PROBA("lector dimensiones",    proba_lector_dimensiones());

    fprintf(
        stderr, "\n=== Summa: %d/%d rectae",
        probationes_rectae, probationes_totae
    );
    if (probationes_falsae > 0)
        fprintf(stderr, ", %d FALSAE", probationes_falsae);
    fprintf(stderr, " ===\n");

    return probationes_falsae > 0 ? 1 : 0;
}
