/*
 * gif.c — inscriptor GIF animatus
 * =================================
 *
 * ARGB8888 tabulas in GIF89a plicam scribit.
 * Quantisationem et perturbationem ad modulos delegat.
 * LZW compressione utitur.
 */

#include "phantasma.h"
#include "quantisatio.h"
#include "perturbatio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * constantiae
 * ================================================================ */

#define PALETA_MAG      128     /* numerus colorum in paleta */
#define PALETA_POTENTIA   7     /* 2^7 = 128 */
#define LZW_MIN_MAG      7     /* minimum code size (2^7 entries) */
#define LZW_MAX_CODEX  4096
#define LZW_HASH_MAG   5003    /* primus > 4096 */

/* ================================================================
 * structura GIF
 * ================================================================ */

struct pfr_gif {
    FILE *plica;
    int lat_fons, alt_fons;
    int lat, alt;
    int scala;
    int mora_cs;
    uint8_t paleta[256][3];
    int paleta_parata;
    int numerus;
    int modus_quant;
    int modus_dither;
};

/* ================================================================
 * LZW compressor
 * ================================================================ */

typedef struct {
    FILE *plica;
    int mag_codis;          /* magnitudo codis currentis (bits) */
    int cod_purgandi;       /* codex purgationis (clear code) */
    int cod_finis;          /* codex finis (EOI) */
    int prox_codex;         /* proximus codex praesto */

    /* tabula hash pro dictionario */
    int16_t hash_praef[LZW_HASH_MAG];
    uint8_t hash_suff[LZW_HASH_MAG];
    int16_t hash_codex[LZW_HASH_MAG];

    /* bit packing (LSB primum) */
    uint32_t bit_alveus;
    int bit_numerus;

    /* sub-block alveus */
    uint8_t sub_alveus[255];
    int sub_pos;
} lzw_t;

static void lzw_sub_emitte(lzw_t *s)
{
    if (s->sub_pos > 0) {
        fputc(s->sub_pos, s->plica);
        fwrite(s->sub_alveus, 1, (size_t)s->sub_pos, s->plica);
        s->sub_pos = 0;
    }
}

static void lzw_byte_emitte(lzw_t *s, uint8_t b)
{
    s->sub_alveus[s->sub_pos++] = b;
    if (s->sub_pos == 255)
        lzw_sub_emitte(s);
}

static void lzw_codicem_emitte(lzw_t *s, int codex)
{
    s->bit_alveus |= (uint32_t)codex << s->bit_numerus;
    s->bit_numerus += s->mag_codis;
    while (s->bit_numerus >= 8) {
        lzw_byte_emitte(s, (uint8_t)(s->bit_alveus & 0xFF));
        s->bit_alveus >>= 8;
        s->bit_numerus -= 8;
    }
}

static void lzw_dict_purga(lzw_t *s)
{
    memset(s->hash_praef, -1, sizeof(s->hash_praef));
    memset(s->hash_codex, -1, sizeof(s->hash_codex));
    s->prox_codex = s->cod_finis + 1;
    s->mag_codis  = LZW_MIN_MAG + 1;
}

static int lzw_quaere(lzw_t *s, int praef, uint8_t suff)
{
    int h = ((praef << 8) ^ suff) % LZW_HASH_MAG;
    while (s->hash_codex[h] >= 0) {
        if (s->hash_praef[h] == praef && s->hash_suff[h] == suff)
            return s->hash_codex[h];
        h = (h + 1) % LZW_HASH_MAG;
    }
    return -1;
}

static void lzw_insere(lzw_t *s, int praef, uint8_t suff)
{
    if (s->prox_codex >= LZW_MAX_CODEX)
        return;
    int h = ((praef << 8) ^ suff) % LZW_HASH_MAG;
    while (s->hash_codex[h] >= 0)
        h = (h + 1) % LZW_HASH_MAG;
    s->hash_praef[h] = (int16_t)praef;
    s->hash_suff[h]  = suff;
    s->hash_codex[h] = (int16_t)s->prox_codex;
    s->prox_codex++;

    if (s->prox_codex > (1 << s->mag_codis) && s->mag_codis < 12)
        s->mag_codis++;
}

static void lzw_comprime(FILE *plica, const uint8_t *data, int n)
{
    lzw_t s;
    memset(&s, 0, sizeof(s));
    s.plica        = plica;
    s.cod_purgandi = 1 << LZW_MIN_MAG;
    s.cod_finis    = s.cod_purgandi + 1;
    memset(s.hash_codex, -1, sizeof(s.hash_codex));

    fputc(LZW_MIN_MAG, plica);

    lzw_dict_purga(&s);
    lzw_codicem_emitte(&s, s.cod_purgandi);

    int w = data[0];
    for (int i = 1; i < n; i++) {
        uint8_t k = data[i];
        int wk    = lzw_quaere(&s, w, k);
        if (wk >= 0) {
            w = wk;
        } else {
            lzw_codicem_emitte(&s, w);
            if (s.prox_codex < LZW_MAX_CODEX) {
                lzw_insere(&s, w, k);
            } else {
                lzw_codicem_emitte(&s, s.cod_purgandi);
                lzw_dict_purga(&s);
            }
            w = k;
        }
    }
    lzw_codicem_emitte(&s, w);
    lzw_codicem_emitte(&s, s.cod_finis);

    if (s.bit_numerus > 0)
        lzw_byte_emitte(&s, (uint8_t)(s.bit_alveus & 0xFF));
    lzw_sub_emitte(&s);

    fputc(0, plica);
}

/* ================================================================
 * auxiliares IO
 * ================================================================ */

static void scribe_u16le(FILE *f, uint16_t v)
{
    fputc(v & 0xFF, f);
    fputc((v >> 8) & 0xFF, f);
}

/* ================================================================
 * interfacies publica
 * ================================================================ */

pfr_gif_t *pfr_gif_initia(
    const char *via, int lat, int alt,
    int mora_cs, int scala
) {
    if (scala < 1)
        scala = 1;

    pfr_gif_t *g = (pfr_gif_t *)calloc(1, sizeof(*g));
    if (!g)
        return NULL;
    g->lat_fons = lat;
    g->alt_fons = alt;
    g->scala    = scala;
    g->lat      = lat / scala;
    g->alt      = alt / scala;
    g->mora_cs  = mora_cs;

    g->plica = fopen(via, "wb");
    if (!g->plica) {
        free(g);
        return NULL;
    }

    return g;
}

void pfr_gif_modum_pone(pfr_gif_t *g, pfr_quant_t quant, pfr_dither_t dither)
{
    if (!g)
        return;
    g->modus_quant  = (int)quant;
    g->modus_dither = (int)dither;
}

static void gif_caput_scribe(pfr_gif_t *g)
{
    FILE *f = g->plica;

    /* GIF89a */
    fwrite("GIF89a", 1, 6, f);
    scribe_u16le(f, (uint16_t)g->lat);
    scribe_u16le(f, (uint16_t)g->alt);

    /* packed: GCT=1, resolutio_coloris=7, sort=0, mag_gct=PALETA_POTENTIA-1 */
    fputc(0x80 | (6 << 4) | (PALETA_POTENTIA - 1), f);
    fputc(0, f);    /* color fundamenti */
    fputc(0, f);    /* ratio aspectus pixeli */

    /* Tabula Colorum Globalis — 2^PALETA_POTENTIA introitus */
    int n_introituum = 1 << PALETA_POTENTIA;
    for (int i = 0; i < n_introituum; i++) {
        if (i < PALETA_MAG) {
            fputc(g->paleta[i][0], f);
            fputc(g->paleta[i][1], f);
            fputc(g->paleta[i][2], f);
        } else {
            fputc(0, f);
            fputc(0, f);
            fputc(0, f);
        }
    }

    /* Extensio Applicationis Netscape — circulus infinitus */
    fputc(0x21, f);
    fputc(0xFF, f);
    fputc(11, f);
    fwrite("NETSCAPE2.0", 1, 11, f);
    fputc(3, f);
    fputc(1, f);
    scribe_u16le(f, 0); /* numerus circulorum: 0 = infinitus */
    fputc(0, f);
}

int pfr_gif_tabulam_adde(pfr_gif_t *g, const uint32_t *pixels)
{
    if (!g || !pixels)
        return -1;

    int n_pix = g->lat * g->alt;

    /* scala et converte ARGB -> RGB */
    uint8_t *rgb = (uint8_t *)malloc((size_t)n_pix * 3);
    if (!rgb)
        return -1;

    for (int y = 0; y < g->alt; y++) {
        for (int x = 0; x < g->lat; x++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            int cnt   = 0;
            for (int sy = 0; sy < g->scala; sy++) {
                int fy = y * g->scala + sy;
                if (fy >= g->alt_fons)
                    continue;
                for (int sx = 0; sx < g->scala; sx++) {
                    int fx = x * g->scala + sx;
                    if (fx >= g->lat_fons)
                        continue;
                    uint32_t px = pixels[fy * g->lat_fons + fx];
                    r_sum += (px >> 16) & 0xFF;
                    g_sum += (px >> 8)  & 0xFF;
                    b_sum +=  px        & 0xFF;
                    cnt++;
                }
            }
            if (cnt > 0) {
                int idx      = (y * g->lat + x) * 3;
                rgb[idx + 0] = (uint8_t)(r_sum / cnt);
                rgb[idx + 1] = (uint8_t)(g_sum / cnt);
                rgb[idx + 2] = (uint8_t)(b_sum / cnt);
            }
        }
    }

    /* genera paletam ex prima tabula */
    if (!g->paleta_parata) {
        switch (g->modus_quant) {
        case PFR_QUANT_OCTARBORIS:
            paletam_genera_octarboris(rgb, n_pix, g->paleta, PALETA_MAG);
            break;
        case PFR_QUANT_KMEDIA:
            paletam_genera_kmedia(rgb, n_pix, g->paleta, PALETA_MAG);
            break;
        default:
            paletam_genera(rgb, n_pix, g->paleta, PALETA_MAG);
            break;
        }
        g->paleta_parata = 1;
        gif_caput_scribe(g);
    }

    /* converte in indices paletae */
    uint8_t *indices = (uint8_t *)malloc((size_t)n_pix);
    if (!indices) {
        free(rgb);
        return -1;
    }

    switch (g->modus_dither) {
    case PFR_DITHER_FLOYD:
        indices_floyd(rgb, g->lat, g->alt, g->paleta, PALETA_MAG, indices);
        break;
    case PFR_DITHER_NULLUM:
        indices_nullum(rgb, g->lat, g->alt, g->paleta, PALETA_MAG, indices);
        break;
    default:
        indices_bayer(rgb, g->lat, g->alt, g->paleta, PALETA_MAG, indices);
        break;
    }

    free(rgb);

    FILE *f = g->plica;

    /* Extensio Moderationis Graphicae */
    fputc(0x21, f);
    fputc(0xF9, f);
    fputc(4, f);
    fputc(0x08, f);     /* dispositio = 2 (restitue ad fundum), sine pelluciditate */
    scribe_u16le(f, (uint16_t)g->mora_cs);
    fputc(0, f);        /* index coloris pellucidi */
    fputc(0, f);        /* terminator */

    /* Descriptor Imaginis */
    fputc(0x2C, f);
    scribe_u16le(f, 0);                    /* sinistrum */
    scribe_u16le(f, 0);                    /* summum */
    scribe_u16le(f, (uint16_t)g->lat);     /* latitudo */
    scribe_u16le(f, (uint16_t)g->alt);     /* altitudo */
    fputc(0, f);                            /* packed: sine TCL */

    /* data compressa LZW */
    lzw_comprime(f, indices, n_pix);

    free(indices);
    g->numerus++;
    return 0;
}

void pfr_gif_fini(pfr_gif_t *g)
{
    if (!g)
        return;
    if (g->plica) {
        fputc(0x3B, g->plica);     /* terminatio GIF */
        fclose(g->plica);
    }
    free(g);
}
