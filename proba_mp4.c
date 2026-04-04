/*
 * proba_mp4.c — probationes unitariae inscriptoris MP4
 *
 * Compilatio:
 *   cc -Wall -Wextra -pedantic -std=c99 -O2 -c pfr_mp4.c -o pfr_mp4.o
 *   cc -Wall -Wextra -pedantic -std=c99 -O2 proba_mp4.c pfr_mp4.o -o proba_mp4
 *
 * Usus:
 *   ./proba_mp4
 */

#include "phantasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* via temporaria pro plicis probationis */
static char via_tmp[256];
static int via_index = 0;

static const char *viam_novam(void)
{
    snprintf(via_tmp, sizeof(via_tmp), "/tmp/proba_mp4_%d.mp4", via_index++);
    return via_tmp;
}

static void plicam_dele(const char *via)
{
    remove(via);
}

/* alveum pixelorum ARGB8888 creat */
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

/* magnitudinem plicae reddit */
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

/* legit 4 bytes big-endian ex plica */
static uint32_t lege_u32(FILE *f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4)
        return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
        ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

/* capsam MP4 quaerit per nomen (4 characteres) */
static int capsam_inveni(
    const char *via, const char *nomen,
    long *pos_out, uint32_t *mag_out
) {
    FILE *f = fopen(via, "rb");
    if (!f)
        return 0;
    fseek(f, 0, SEEK_END);
    long finis = ftell(f);
    fseek(f, 0, SEEK_SET);

    while (ftell(f) < finis - 8) {
        long p       = ftell(f);
        uint32_t mag = lege_u32(f);
        char typ[5] = {0};
        if (fread(typ, 1, 4, f) != 4)
            break;
        if (mag < 8)
            break;

        if (memcmp(typ, nomen, 4) == 0) {
            if (pos_out)
                *pos_out = p;
            if (mag_out)
                *mag_out = mag;
            fclose(f);
            return 1;
        }
        fseek(f, p + (long)mag, SEEK_SET);
    }
    fclose(f);
    return 0;
}

/* ================================================================
 * probationes
 * ================================================================ */

/* --- parametra nulla --- */

static const char *proba_initia_via_nulla(void)
{
    pfr_mp4_t *m = pfr_mp4_initia(NULL, 64, 64, 30);
    if (m) {
        pfr_mp4_fini(m);
        return "non debet initiari cum via nulla";
    }
    return NULL;
}

static const char *proba_adde_nulla(void)
{
    if (pfr_mp4_tabulam_adde(NULL, NULL) != -1)
        return "debet -1 reddere cum inscriptore nullo";
    return NULL;
}

static const char *proba_fini_nulla(void)
{
    /* non debet exitium facere */
    pfr_mp4_fini(NULL);
    return NULL;
}

/* --- vita basica --- */

static const char *proba_vita_simplex(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 64, 64, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(64, 64, 0xFF804020);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    if (res != 0) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    if (mag <= 0) {
        plicam_dele(via);
        return "plica vacua";
    }

    plicam_dele(via);
    return NULL;
}

static const char *proba_plures_tabulae(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 128, 96, 24);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(128, 96, 0xFFFF0000);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    for (int i = 0; i < 10; i++) {
        int res = pfr_mp4_tabulam_adde(m, pix);
        if (res != 0) {
            free(pix);
            pfr_mp4_fini(m);
            plicam_dele(via);
            return "adde falsum";
        }
    }

    free(pix);
    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- sine tabulis --- */

static const char *proba_sine_tabulis(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 32, 32, 30);
    if (!m)
        return "initia falsum";

    /* fini sine ulla tabula addita */
    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    /* plica debet existere (saltem ftyp + mdat caput) */
    if (mag <= 0)
        return "plica non scripta";
    return NULL;
}

/* --- dimensiones non-16-alignatae --- */

static const char *proba_dimensiones_impares(void)
{
    const char *via = viam_novam();
    /* 100x75 — non multiplex 16 */
    pfr_mp4_t *m = pfr_mp4_initia(via, 100, 75, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(100, 75, 0xFF00FF00);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    if (res != 0) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- dimensiones minimae --- */

static const char *proba_dimensio_1x1(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 1, 1, 30);
    if (!m)
        return "initia falsum";

    uint32_t pix = 0xFFFFFFFF;
    int res      = pfr_mp4_tabulam_adde(m, &pix);
    if (res != 0) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

static const char *proba_dimensio_2x2(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 2, 2, 30);
    if (!m)
        return "initia falsum";

    uint32_t pix[4] = {0xFF000000, 0xFFFF0000, 0xFF00FF00, 0xFF0000FF};
    int res = pfr_mp4_tabulam_adde(m, pix);
    if (res != 0) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "adde falsum";
    }

    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- dimensiones 768x768 (sicut excrashy.c) --- */

static const char *proba_dimensio_768(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 768, 768, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(768, 768, 0xFF402010);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    /* tres tabulae sicut in inscriptione brevi */
    for (int i = 0; i < 3; i++) {
        /* muta colorem paululum */
        for (int j = 0; j < 768 * 768; j++)
            pix[j] = 0xFF000000 | (uint32_t)((i * 30 + j) & 0xFFFFFF);

        int res = pfr_mp4_tabulam_adde(m, pix);
        if (res != 0) {
            free(pix);
            pfr_mp4_fini(m);
            plicam_dele(via);
            return "adde falsum";
        }
    }

    free(pix);
    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- structura MP4 valida --- */

static const char *proba_capsae_mp4(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 64, 48, 25);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(64, 48, 0xFFAA5533);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    pfr_mp4_tabulam_adde(m, pix);
    pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    pfr_mp4_fini(m);

    /* verifica capsas fundamentales */
    int ftyp_inv = capsam_inveni(via, "ftyp", NULL, NULL);
    int mdat_inv = capsam_inveni(via, "mdat", NULL, NULL);
    int moov_inv = capsam_inveni(via, "moov", NULL, NULL);

    plicam_dele(via);

    if (!ftyp_inv)
        return "capsa ftyp non inventa";
    if (!mdat_inv)
        return "capsa mdat non inventa";
    if (!moov_inv)
        return "capsa moov non inventa";
    return NULL;
}

/* --- ftyp contenta --- */

static const char *proba_ftyp_contenta(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 32, 32, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(32, 32, 0xFF000000);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }
    pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    pfr_mp4_fini(m);

    FILE *f = fopen(via, "rb");
    plicam_dele(via);
    if (!f)
        return "plica non aperitur";

    /* ftyp debet esse prima capsa */
    (void)lege_u32(f);  /* praetermitte magnitudinem */
    char typ[5] = {0};
    if (fread(typ, 1, 4, f) != 4) {
        fclose(f);
        return "lectura falli";
    }
    if (memcmp(typ, "ftyp", 4) != 0) {
        fclose(f);
        return "ftyp non prima";
    }

    char marca[5] = {0};
    if (fread(marca, 1, 4, f) != 4) {
        fclose(f);
        return "marca legi non potest";
    }
    if (memcmp(marca, "isom", 4) != 0) {
        fclose(f);
        return "marca non isom";
    }

    fclose(f);
    return NULL;
}

/* --- pixels nullus --- */

static const char *proba_pixels_nullus(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 64, 64, 30);
    if (!m)
        return "initia falsum";

    int res = pfr_mp4_tabulam_adde(m, NULL);
    pfr_mp4_fini(m);
    plicam_dele(via);

    if (res != -1)
        return "debet -1 reddere cum pixelibus nullis";
    return NULL;
}

/* --- multae tabulae cum coloribus diversis --- */

static const char *proba_gradiens(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 320, 240, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = (uint32_t *)malloc(320 * 240 * sizeof(uint32_t));
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    for (int t = 0; t < 5; t++) {
        /* gradiens diversus pro unaquaque tabula */
        for (int y = 0; y < 240; y++) {
            for (int x = 0; x < 320; x++) {
                uint8_t r = (uint8_t)((x + t * 20) & 0xFF);
                uint8_t g = (uint8_t)((y + t * 30) & 0xFF);
                uint8_t b = (uint8_t)((x + y + t * 10) & 0xFF);
                pix[y * 320 + x] = 0xFF000000 | ((uint32_t)r << 16) |
                    ((uint32_t)g << 8) | (uint32_t)b;
            }
        }
        int res = pfr_mp4_tabulam_adde(m, pix);
        if (res != 0) {
            free(pix);
            pfr_mp4_fini(m);
            plicam_dele(via);
            return "adde falsum";
        }
    }

    free(pix);
    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- mdat magnitudo congruens --- */

static const char *proba_mdat_magnitudo(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 16, 16, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(16, 16, 0xFF808080);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    pfr_mp4_fini(m);

    /* mdat magnitudo debet maior quam 8 esse (caput + data) */
    uint32_t mdat_mag = 0;
    int inv = capsam_inveni(via, "mdat", NULL, &mdat_mag);
    plicam_dele(via);

    if (!inv)
        return "mdat non inventa";
    if (mdat_mag <= 8)
        return "mdat nimis parva";

    /* pro 16x16 cum I_PCM: 1 MB × (256 Y + 64 Cb + 64 Cr) + caput */
    if (mdat_mag < 400)
        return "mdat nimis parva pro 16x16";

    return NULL;
}

/* --- dimensiones impares singulae (latitudo impar) --- */

static const char *proba_latitudo_impar(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 17, 16, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(17, 16, 0xFFAABBCC);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    pfr_mp4_fini(m);
    plicam_dele(via);

    if (res != 0)
        return "adde falsum cum latitudine impari";
    return NULL;
}

/* --- altitudo impar --- */

static const char *proba_altitudo_impar(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 16, 17, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(16, 17, 0xFFDDEEFF);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    int res = pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    pfr_mp4_fini(m);
    plicam_dele(via);

    if (res != 0)
        return "adde falsum cum altitudine impari";
    return NULL;
}

/* --- FPS = 1 (lentissimum) --- */

static const char *proba_fps_unum(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 32, 32, 1);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(32, 32, 0xFF112233);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    pfr_mp4_fini(m);

    int moov = capsam_inveni(via, "moov", NULL, NULL);
    plicam_dele(via);

    if (!moov)
        return "moov non inventa cum fps=1";
    return NULL;
}

/* --- multae tabulae parvae (17 tabulae, ultra frame_num wrap) --- */

static const char *proba_frame_num_wrap(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 16, 16, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(16, 16, 0xFF000000);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }

    /* 17 tabulae — transit frame_num maximum (16) */
    for (int i = 0; i < 17; i++) {
        pix[0]  = 0xFF000000 | (uint32_t)i;
        int res = pfr_mp4_tabulam_adde(m, pix);
        if (res != 0) {
            free(pix);
            pfr_mp4_fini(m);
            plicam_dele(via);
            return "adde falsum";
        }
    }

    free(pix);
    pfr_mp4_fini(m);

    long mag = plica_magnitudo(via);
    plicam_dele(via);
    if (mag <= 0)
        return "plica vacua";
    return NULL;
}

/* --- dimensio 3x3 (valde parva, impar) --- */

static const char *proba_dimensio_3x3(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 3, 3, 30);
    if (!m)
        return "initia falsum";

    uint32_t pix[9];
    for (int i = 0; i < 9; i++)
        pix[i] = 0xFFFF00FF;

    int res = pfr_mp4_tabulam_adde(m, pix);
    pfr_mp4_fini(m);
    plicam_dele(via);

    if (res != 0)
        return "adde falsum cum 3x3";
    return NULL;
}

/* --- ordo capsarum: ftyp, mdat, moov --- */

static const char *proba_ordo_capsarum(void)
{
    const char *via = viam_novam();
    pfr_mp4_t *m    = pfr_mp4_initia(via, 32, 32, 30);
    if (!m)
        return "initia falsum";

    uint32_t *pix = pixels_crea(32, 32, 0xFFAAAAAA);
    if (!pix) {
        pfr_mp4_fini(m);
        plicam_dele(via);
        return "memoria";
    }
    pfr_mp4_tabulam_adde(m, pix);
    free(pix);
    pfr_mp4_fini(m);

    long pos_ftyp = -1, pos_mdat = -1, pos_moov = -1;
    capsam_inveni(via, "ftyp", &pos_ftyp, NULL);
    capsam_inveni(via, "mdat", &pos_mdat, NULL);
    capsam_inveni(via, "moov", &pos_moov, NULL);
    plicam_dele(via);

    if (pos_ftyp < 0 || pos_mdat < 0 || pos_moov < 0)
        return "capsae non inventae";
    if (pos_ftyp >= pos_mdat)
        return "ftyp debet ante mdat esse";
    if (pos_mdat >= pos_moov)
        return "mdat debet ante moov esse";
    return NULL;
}

/* ================================================================
 * principium
 * ================================================================ */

int main(void)
{
    fprintf(stderr, "=== Probationes inscriptoris MP4 ===\n\n");

    fprintf(stderr, "— parametra nulla —\n");
    PROBA("initia via nulla",     proba_initia_via_nulla());
    PROBA("adde inscriptor nullus", proba_adde_nulla());
    PROBA("fini nullus",          proba_fini_nulla());
    PROBA("pixels nullus",        proba_pixels_nullus());

    fprintf(stderr, "\n— vita basica —\n");
    PROBA("vita simplex",         proba_vita_simplex());
    PROBA("plures tabulae",       proba_plures_tabulae());
    PROBA("sine tabulis",         proba_sine_tabulis());
    PROBA("gradiens",             proba_gradiens());

    fprintf(stderr, "\n— dimensiones —\n");
    PROBA("1x1",                  proba_dimensio_1x1());
    PROBA("2x2",                  proba_dimensio_2x2());
    PROBA("3x3",                  proba_dimensio_3x3());
    PROBA("dimensiones impares",  proba_dimensiones_impares());
    PROBA("latitudo impar",       proba_latitudo_impar());
    PROBA("altitudo impar",       proba_altitudo_impar());
    PROBA("768x768 (excrashy)",   proba_dimensio_768());

    fprintf(stderr, "\n— structura MP4 —\n");
    PROBA("capsae ftyp/mdat/moov", proba_capsae_mp4());
    PROBA("ftyp contenta",        proba_ftyp_contenta());
    PROBA("mdat magnitudo",       proba_mdat_magnitudo());
    PROBA("ordo capsarum",        proba_ordo_capsarum());

    fprintf(stderr, "\n— limites —\n");
    PROBA("fps unum",             proba_fps_unum());
    PROBA("frame_num wrap (17)",  proba_frame_num_wrap());

    fprintf(
        stderr, "\n=== Summa: %d/%d rectae",
        probationes_rectae, probationes_totae
    );
    if (probationes_falsae > 0)
        fprintf(stderr, ", %d FALSAE", probationes_falsae);
    fprintf(stderr, " ===\n");

    return probationes_falsae > 0 ? 1 : 0;
}
