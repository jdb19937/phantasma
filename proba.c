/*
 * proba.c — probatio bibliothecae PFR
 *
 * Fenestram aperit, circulum coloratum pingit qui
 * clavibus et rota muris regitur.
 *
 * Claves:
 *   Sagittae / WASD  — circulum movere
 *   +/-              — radium mutare
 *   Spatium          — colorem mutare
 *   Rota muris       — radium mutare
 *   Q / Escape       — exire
 */

#include "phantasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LATITUDO  640
#define ALTITUDO  480

static void circulum_pinge(uint32_t *alveus, int lat, int alt,
                            int cx, int cy, int radius,
                            uint8_t r, uint8_t g, uint8_t b)
{
    int r2 = radius * radius;
    for (int y = cy - radius; y <= cy + radius; y++) {
        if (y < 0 || y >= alt) continue;
        for (int x = cx - radius; x <= cx + radius; x++) {
            if (x < 0 || x >= lat) continue;
            int dx = x - cx;
            int dy = y - cy;
            int d2 = dx * dx + dy * dy;
            if (d2 <= r2) {
                /* gradiens ab centro */
                double f = 1.0 - (double)d2 / (double)r2;
                uint8_t pr = (uint8_t)(r * f);
                uint8_t pg = (uint8_t)(g * f);
                uint8_t pb = (uint8_t)(b * f);
                alveus[y * lat + x] = (0xFFu << 24)
                    | ((uint32_t)pr << 16)
                    | ((uint32_t)pg << 8)
                    |  (uint32_t)pb;
            }
        }
    }
}

static void fundum_pinge(uint32_t *alveus, int lat, int alt, pfr_u32 tempus)
{
    for (int y = 0; y < alt; y++) {
        for (int x = 0; x < lat; x++) {
            /* cancelli lenti */
            int cx = (x + (int)(tempus / 40)) % 64;
            int cy = (y + (int)(tempus / 60)) % 64;
            int in_cancello = (cx < 32) ^ (cy < 32);
            uint8_t v = in_cancello ? 30 : 20;
            alveus[y * lat + x] = (0xFFu << 24)
                | ((uint32_t)v << 16)
                | ((uint32_t)v << 8)
                |  (uint32_t)v;
        }
    }
}

int main(void)
{
    if (pfr_initia(PFR_INITIA_VIDEO) != 0) {
        fprintf(stderr, "ERRATUM: pfr_initia: %s\n", pfr_erratum());
        return 1;
    }

    pfr_fenestra_t *fenestra = pfr_fenestram_crea(
        "PFR Probatio", PFR_POS_MEDIUM, PFR_POS_MEDIUM,
        LATITUDO, ALTITUDO, 0);
    if (!fenestra) {
        fprintf(stderr, "ERRATUM: fenestra: %s\n", pfr_erratum());
        pfr_fini();
        return 1;
    }

    pfr_pictor_t *pictor = pfr_pictorem_crea(fenestra, -1,
        PFR_PICTOR_CELER | PFR_PICTOR_SYNC);
    if (!pictor) {
        fprintf(stderr, "ERRATUM: pictor\n");
        pfr_fenestram_destrue(fenestra);
        pfr_fini();
        return 1;
    }

    pfr_textura_t *textura = pfr_texturam_crea(pictor,
        PFR_PIXEL_ARGB8888, PFR_TEXTURA_FLUENS, LATITUDO, ALTITUDO);
    if (!textura) {
        fprintf(stderr, "ERRATUM: textura\n");
        pfr_pictorem_destrue(pictor);
        pfr_fenestram_destrue(fenestra);
        pfr_fini();
        return 1;
    }

    uint32_t *imago = (uint32_t *)malloc(
        (size_t)LATITUDO * ALTITUDO * sizeof(uint32_t));

    int cx = LATITUDO / 2;
    int cy = ALTITUDO / 2;
    int radius = 60;
    int color_index = 0;

    static const uint8_t colores[][3] = {
        {255, 100,  50},   /* aurantius */
        { 50, 200, 255},   /* caeruleus */
        {100, 255, 100},   /* viridis */
        {255, 255,  50},   /* flavus */
        {255,  80, 200},   /* roseus */
    };
    int num_colores = sizeof(colores) / sizeof(colores[0]);

    fprintf(stderr, "PFR probatio currit.\n");
    fprintf(stderr, "  Sagittae/WASD — move  +/- — radius  Spatium — color\n");
    fprintf(stderr, "  Rota muris — radius   Q/Esc — exi\n");

    int currit = 1;
    pfr_u32 tempus_prius = pfr_tempus();
    int tabulae = 0;

    while (currit) {
        pfr_u32 nunc = pfr_tempus();
        double dt = (double)(nunc - tempus_prius) / 1000.0;
        tempus_prius = nunc;

        pfr_eventus_t ev;
        while (pfr_eventum_lege(&ev)) {
            if (ev.typus == PFR_EXITUS) {
                currit = 0;
            } else if (ev.typus == PFR_ROTA_MURIS) {
                radius += ev.rota.y * 5;
                if (radius < 5)   radius = 5;
                if (radius > 200) radius = 200;
            } else if (ev.typus == PFR_CLAVIS_INF) {
                switch (ev.clavis.signum.symbolum) {
                case PFR_CL_EFFUGIUM:
                case 'q':
                    currit = 0;
                    break;
                case PFR_CL_SPATIUM:
                    color_index = (color_index + 1) % num_colores;
                    break;
                case PFR_CL_AEQUALE:
                case PFR_CL_PLUS:
                    radius += 5;
                    if (radius > 200) radius = 200;
                    break;
                case PFR_CL_MINUS:
                    radius -= 5;
                    if (radius < 5) radius = 5;
                    break;
                }
            }
        }

        /* motus continuus per claves depressas */
        const pfr_u8 *claves = pfr_claves_status(NULL);
        int celer = (int)(300.0 * dt);
        if (celer < 1) celer = 1;
        if (claves[PFR_SC_SINISTRUM] || claves[PFR_SC_A]) cx -= celer;
        if (claves[PFR_SC_DEXTRUM]   || claves[PFR_SC_D]) cx += celer;
        if (claves[PFR_SC_SURSUM]    || claves[PFR_SC_W]) cy -= celer;
        if (claves[PFR_SC_DEORSUM]   || claves[PFR_SC_S]) cy += celer;

        if (cx < 0) cx = 0;
        if (cx >= LATITUDO) cx = LATITUDO - 1;
        if (cy < 0) cy = 0;
        if (cy >= ALTITUDO) cy = ALTITUDO - 1;

        /* pinge */
        fundum_pinge(imago, LATITUDO, ALTITUDO, nunc);
        circulum_pinge(imago, LATITUDO, ALTITUDO, cx, cy, radius,
                        colores[color_index][0],
                        colores[color_index][1],
                        colores[color_index][2]);

        pfr_texturam_renova(textura, NULL, imago, LATITUDO * 4);
        pfr_purga(pictor);
        pfr_texturam_pinge(pictor, textura, NULL, NULL);

        /* indicium: rectum album in angulo */
        pfr_colorem_pone(pictor, 255, 255, 255, 255);
        pfr_rectum_t ind = {8, 8, 12, 12};
        pfr_rectum_imple(pictor, &ind);

        pfr_praesenta(pictor);
        tabulae++;

        /* FPS ad stderr omni secundo */
        if (tabulae % 60 == 0)
            fprintf(stderr, "\r  t=%u ms  pos=(%d,%d)  r=%d  ",
                    nunc, cx, cy, radius);

        pfr_pausa(16);
    }

    fprintf(stderr, "\nVale. (%d tabulae)\n", tabulae);

    free(imago);
    pfr_texturam_destrue(textura);
    pfr_pictorem_destrue(pictor);
    pfr_fenestram_destrue(fenestra);
    pfr_fini();

    return 0;
}
