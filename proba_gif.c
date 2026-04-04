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
    PROBA("celeritas (5×384x384)", proba_celeritas());

    fprintf(
        stderr, "\n=== Summa: %d/%d rectae",
        probationes_rectae, probationes_totae
    );
    if (probationes_falsae > 0)
        fprintf(stderr, ", %d FALSAE", probationes_falsae);
    fprintf(stderr, " ===\n");

    return probationes_falsae > 0 ? 1 : 0;
}
