/*
 * gif_lector.c — lector GIF (decoder)
 * =====================================
 *
 * GIF87a/GIF89a plicas legit.
 * Tabulas in ARGB8888 pixelos convertit.
 * Pelluciditatem per GCE extensionem sustinet.
 * LZW decompressione utitur.
 */

#include "phantasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * constantiae
 * ================================================================ */

#define LZW_MAX_CODEX  4096
#define LZW_MAX_ALVEUS 4096

/* ================================================================
 * structura lectoris
 * ================================================================ */

struct pfr_gif_lector {
    FILE *plica;
    int lat, alt;
    int habet_gct;
    int mag_gct;
    uint8_t gct[256][3];
    int circuli;

    /* status tabulae currentis */
    int tab_lat, tab_alt;
    int tab_sin, tab_sum;
    int habet_lct;
    int mag_lct;
    uint8_t lct[256][3];

    /* GCE */
    int habet_gce;
    int gce_dispositio;
    int gce_pellucidus;
    int gce_index_pelluc;
    int gce_mora_cs;

    /* canvas compositus */
    uint32_t *canvas;
};

/* ================================================================
 * auxiliares IO
 * ================================================================ */

static uint16_t lege_u16le(FILE *f)
{
    int lo = fgetc(f);
    int hi = fgetc(f);
    return (uint16_t)((hi << 8) | lo);
}

static void sub_blocos_salta(FILE *f)
{
    for (;;) {
        int mag = fgetc(f);
        if (mag <= 0)
            break;
        fseek(f, mag, SEEK_CUR);
    }
}

/* ================================================================
 * LZW decompressor
 * ================================================================ */

typedef struct {
    int praef[LZW_MAX_CODEX];
    uint8_t suff[LZW_MAX_CODEX];
    int longitudo[LZW_MAX_CODEX];
    int mag_init;
    int cod_purgandi;
    int cod_finis;
    int prox_codex;
    int mag_codis;
} lzw_dict_t;

typedef struct {
    FILE *plica;
    uint32_t bit_alveus;
    int bit_numerus;
    uint8_t sub_alveus[255];
    int sub_pos;
    int sub_mag;
    int finis;
} lzw_fons_t;

static void lzw_fontem_initia(lzw_fons_t *f, FILE *plica)
{
    f->plica       = plica;
    f->bit_alveus  = 0;
    f->bit_numerus = 0;
    f->sub_pos     = 0;
    f->sub_mag     = 0;
    f->finis       = 0;
}

static int lzw_byte_lege(lzw_fons_t *f)
{
    if (f->sub_pos >= f->sub_mag) {
        int mag = fgetc(f->plica);
        if (mag <= 0) {
            f->finis = 1;
            return -1;
        }
        f->sub_mag = mag;
        f->sub_pos = 0;
        if (fread(f->sub_alveus, 1, (size_t)mag, f->plica) != (size_t)mag) {
            f->finis = 1;
            return -1;
        }
    }
    return f->sub_alveus[f->sub_pos++];
}

static int lzw_codicem_lege(lzw_fons_t *f, int mag_codis)
{
    while (f->bit_numerus < mag_codis) {
        int b = lzw_byte_lege(f);
        if (b < 0)
            return -1;
        f->bit_alveus |= (uint32_t)b << f->bit_numerus;
        f->bit_numerus += 8;
    }
    int codex = (int)(f->bit_alveus & ((1u << mag_codis) - 1));
    f->bit_alveus >>= mag_codis;
    f->bit_numerus -= mag_codis;
    return codex;
}

static void lzw_dict_purga(lzw_dict_t *d)
{
    int n = 1 << d->mag_init;
    for (int i = 0; i < n; i++) {
        d->praef[i]     = -1;
        d->suff[i]      = (uint8_t)i;
        d->longitudo[i] = 1;
    }
    d->cod_purgandi = n;
    d->cod_finis    = n + 1;
    d->prox_codex   = n + 2;
    d->mag_codis    = d->mag_init + 1;
}

/* emitte catenam codicis in alveum, redit numerum byte scriptorum */
static int lzw_catenam_emitte(
    lzw_dict_t *d, int codex,
    uint8_t *alveus, int max_n
) {
    if (codex < 0 || codex >= d->prox_codex)
        return 0;
    int longitudo = d->longitudo[codex];
    if (longitudo > max_n)
        return 0;
    int pos = longitudo - 1;
    int c   = codex;
    while (c >= 0 && pos >= 0) {
        alveus[pos--] = d->suff[c];
        c = d->praef[c];
    }
    return longitudo;
}

static int lzw_decomprime(
    FILE *plica, int mag_min,
    uint8_t *data, int max_n
) {
    lzw_dict_t d;
    d.mag_init = mag_min;
    lzw_dict_purga(&d);

    lzw_fons_t fons;
    lzw_fontem_initia(&fons, plica);

    uint8_t alveus[LZW_MAX_ALVEUS];
    int pos   = 0;
    int prior = -1;

    for (;;) {
        int codex = lzw_codicem_lege(&fons, d.mag_codis);
        if (codex < 0 || codex == d.cod_finis)
            break;

        if (codex == d.cod_purgandi) {
            lzw_dict_purga(&d);
            prior = -1;
            continue;
        }

        if (codex < d.prox_codex) {
            int n = lzw_catenam_emitte(&d, codex, alveus, LZW_MAX_ALVEUS);
            for (int i = 0; i < n && pos < max_n; i++)
                data[pos++] = alveus[i];

            if (prior >= 0 && d.prox_codex < LZW_MAX_CODEX) {
                d.praef[d.prox_codex]     = prior;
                d.suff[d.prox_codex]      = alveus[0];
                d.longitudo[d.prox_codex] = d.longitudo[prior] + 1;
                d.prox_codex++;
                if (d.prox_codex > (1 << d.mag_codis) && d.mag_codis < 12)
                    d.mag_codis++;
            }
        } else {
            /* codex == prox_codex: casus specialis */
            if (prior < 0)
                break;
            int n = lzw_catenam_emitte(&d, prior, alveus, LZW_MAX_ALVEUS - 1);
            if (n <= 0)
                break;
            alveus[n] = alveus[0];
            n++;
            for (int i = 0; i < n && pos < max_n; i++)
                data[pos++] = alveus[i];

            if (d.prox_codex < LZW_MAX_CODEX) {
                d.praef[d.prox_codex]     = prior;
                d.suff[d.prox_codex]      = alveus[0];
                d.longitudo[d.prox_codex] = d.longitudo[prior] + 1;
                d.prox_codex++;
                if (d.prox_codex > (1 << d.mag_codis) && d.mag_codis < 12)
                    d.mag_codis++;
            }
        }
        prior = codex;
    }

    /* consume sub-blocos reliquos */
    if (!fons.finis) {
        for (;;) {
            int mag = fgetc(plica);
            if (mag <= 0)
                break;
            fseek(plica, mag, SEEK_CUR);
        }
    }

    return pos;
}

/* ================================================================
 * interfacies publica
 * ================================================================ */

pfr_gif_lector_t *pfr_gif_lege_initia(const char *via)
{
    if (!via)
        return NULL;

    FILE *f = fopen(via, "rb");
    if (!f)
        return NULL;

    /* lege signaculum */
    char signum[6];
    if (fread(signum, 1, 6, f) != 6) {
        fclose(f);
        return NULL;
    }
    if (
        memcmp(signum, "GIF87a", 6) != 0 &&
        memcmp(signum, "GIF89a", 6) != 0
    ) {
        fclose(f);
        return NULL;
    }

    pfr_gif_lector_t *l = (pfr_gif_lector_t *)calloc(1, sizeof(*l));
    if (!l) {
        fclose(f);
        return NULL;
    }
    l->plica = f;

    /* Logical Screen Descriptor */
    l->lat = lege_u16le(f);
    l->alt = lege_u16le(f);

    int packed = fgetc(f);
    fgetc(f);   /* color fundamenti */
    fgetc(f);   /* ratio aspectus */

    l->habet_gct = (packed >> 7) & 1;
    if (l->habet_gct) {
        int potentia = (packed & 0x07) + 1;
        l->mag_gct   = 1 << potentia;
        for (int i = 0; i < l->mag_gct; i++) {
            l->gct[i][0] = (uint8_t)fgetc(f);
            l->gct[i][1] = (uint8_t)fgetc(f);
            l->gct[i][2] = (uint8_t)fgetc(f);
        }
    }

    /* canvas initialis */
    l->canvas = (uint32_t *)calloc(
        (size_t)l->lat * l->alt, sizeof(uint32_t)
    );

    return l;
}

int pfr_gif_lege_dimensiones(pfr_gif_lector_t *l, int *lat, int *alt)
{
    if (!l)
        return -1;
    if (lat)
        *lat = l->lat;
    if (alt)
        *alt = l->alt;
    return 0;
}

int pfr_gif_lege_tabulam(pfr_gif_lector_t *l, uint32_t *pixels)
{
    if (!l || !l->plica)
        return -1;

    FILE *f = l->plica;

    /* reseta GCE */
    l->habet_gce        = 0;
    l->gce_pellucidus   = 0;
    l->gce_dispositio   = 0;
    l->gce_index_pelluc = 0;
    l->gce_mora_cs      = 0;

    /* quaere proximam imaginem */
    for (;;) {
        int intro = fgetc(f);
        if (intro == EOF || intro == 0x3B)
            return -1;   /* finis GIF */

        if (intro == 0x21) {
            /* extensio */
            int label = fgetc(f);
            if (label == 0xF9) {
                /* Graphics Control Extension */
                int mag = fgetc(f);
                if (mag >= 4) {
                    int pack   = fgetc(f);
                    l->gce_mora_cs     = lege_u16le(f);
                    l->gce_index_pelluc = fgetc(f);
                    l->gce_dispositio  = (pack >> 2) & 0x07;
                    l->gce_pellucidus  = pack & 0x01;
                    l->habet_gce       = 1;
                    /* salta reliquos bytes si mag > 4 */
                    for (int i = 4; i < mag; i++)
                        fgetc(f);
                }
                fgetc(f);   /* terminator */
            } else {
                sub_blocos_salta(f);
            }
            continue;
        }

        if (intro == 0x2C) {
            /* Image Descriptor */
            l->tab_sin = lege_u16le(f);
            l->tab_sum = lege_u16le(f);
            l->tab_lat = lege_u16le(f);
            l->tab_alt = lege_u16le(f);

            int packed_img = fgetc(f);
            int interlace  = (packed_img >> 6) & 1;
            l->habet_lct   = (packed_img >> 7) & 1;

            if (l->habet_lct) {
                int pot    = (packed_img & 0x07) + 1;
                l->mag_lct = 1 << pot;
                for (int i = 0; i < l->mag_lct; i++) {
                    l->lct[i][0] = (uint8_t)fgetc(f);
                    l->lct[i][1] = (uint8_t)fgetc(f);
                    l->lct[i][2] = (uint8_t)fgetc(f);
                }
            }

            /* elige paletam */
            uint8_t (*pal)[3] = l->habet_lct ? l->lct : l->gct;
            int n_pal = l->habet_lct ? l->mag_lct :
                (l->habet_gct ? l->mag_gct : 0);

            /* LZW minimum code size */
            int lzw_min = fgetc(f);
            if (lzw_min < 2 || lzw_min > 12) {
                sub_blocos_salta(f);
                continue;
            }

            int n_pix        = l->tab_lat * l->tab_alt;
            uint8_t *indices = (uint8_t *)malloc((size_t)n_pix);
            if (!indices) {
                sub_blocos_salta(f);
                return -1;
            }

            int n_lecta = lzw_decomprime(f, lzw_min, indices, n_pix);
            (void)n_lecta;

            /* dispositio prioris */
            if (l->gce_dispositio == 2) {
                /* restitue ad fundum (pellucidus) */
                for (int y = 0; y < l->tab_alt; y++) {
                    int cy = l->tab_sum + y;
                    if (cy >= l->alt)
                        continue;
                    for (int x = 0; x < l->tab_lat; x++) {
                        int cx = l->tab_sin + x;
                        if (cx >= l->lat)
                            continue;
                        l->canvas[cy * l->lat + cx] = 0x00000000;
                    }
                }
            }

            /* pinge indices in canvas */
            for (int y = 0; y < l->tab_alt; y++) {
                int cy = l->tab_sum + y;
                if (cy >= l->alt)
                    continue;

                /* interlace: computa lineam fontem */
                int sy = y;
                if (interlace) {
                    static const int passus_init[] = {0, 4, 2, 1};
                    static const int passus_inc[]  = {8, 8, 4, 2};
                    int linea = 0;
                    for (int p = 0; p < 4; p++) {
                        int n_in_passu = 0;
                        for (
                            int ll = passus_init[p];
                            ll < l->tab_alt;
                            ll += passus_inc[p]
                        )
                            n_in_passu++;
                        if (y < linea + n_in_passu) {
                            sy = passus_init[p] +
                                (y - linea) * passus_inc[p];
                            break;
                        }
                        linea += n_in_passu;
                    }
                }

                for (int x = 0; x < l->tab_lat; x++) {
                    int cx = l->tab_sin + x;
                    if (cx >= l->lat)
                        continue;

                    int ii  = sy * l->tab_lat + x;
                    int idx = (ii < n_pix) ? indices[ii] : 0;

                    if (
                        l->habet_gce && l->gce_pellucidus &&
                        idx == l->gce_index_pelluc
                    ) {
                        l->canvas[cy * l->lat + cx] = 0x00000000;
                    } else if (idx < n_pal) {
                        l->canvas[cy * l->lat + cx] =
                            0xFF000000 |
                            ((uint32_t)pal[idx][0] << 16) |
                            ((uint32_t)pal[idx][1] << 8) |
                            (uint32_t)pal[idx][2];
                    }
                }
            }

            free(indices);

            /* copia canvas ad output */
            if (pixels)
                memcpy(
                    pixels,
                    l->canvas,
                    (size_t)l->lat * l->alt * sizeof(uint32_t)
                );

            return 0;
        }

        /* ignotum byte — salta */
    }
}

void pfr_gif_lege_fini(pfr_gif_lector_t *l)
{
    if (!l)
        return;
    if (l->plica)
        fclose(l->plica);
    free(l->canvas);
    free(l);
}
