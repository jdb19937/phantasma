/*
 * pfr_gif.c — inscriptor GIF animatus
 * ======================================
 *
 * ARGB8888 tabulas in GIF89a plicam scribit.
 * Colores per median-cut ad 128 reducit.
 * Bayer dithering applicat.
 * LZW compressione utitur.
 */

#include "phantasma.h"

#include <math.h>
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
};

/* ================================================================
 * quantisatio colorum — median cut
 * ================================================================ */

typedef struct {
    int r_min, r_max;
    int g_min, g_max;
    int b_min, b_max;
    int64_t r_sum, g_sum, b_sum;
    int numerus;
} capsa_t;

/* maximum dimensionem capsae reddit */
static int capsa_amplitudo(const capsa_t *c)
{
    int dr = c->r_max - c->r_min;
    int dg = c->g_max - c->g_min;
    int db = c->b_max - c->b_min;
    if (dr >= dg && dr >= db) return 0;
    if (dg >= dr && dg >= db) return 1;
    return 2;
}

static int capsa_magnitudo(const capsa_t *c)
{
    int dr = c->r_max - c->r_min;
    int dg = c->g_max - c->g_min;
    int db = c->b_max - c->b_min;
    if (dr >= dg && dr >= db) return dr;
    if (dg >= dr && dg >= db) return dg;
    return db;
}

static void paletam_genera(const uint8_t *rgb, int n_pix,
                            uint8_t paleta[][3], int n_colorum)
{
    /* capsas initia */
    capsa_t *capsae = (capsa_t *)calloc((size_t)n_colorum, sizeof(capsa_t));
    int n_capsarum = 1;

    capsae[0].r_min = capsae[0].g_min = capsae[0].b_min = 255;
    capsae[0].r_max = capsae[0].g_max = capsae[0].b_max = 0;

    /* limites primae capsae computa */
    for (int i = 0; i < n_pix; i++) {
        int r = rgb[i * 3 + 0];
        int g = rgb[i * 3 + 1];
        int b = rgb[i * 3 + 2];
        if (r < capsae[0].r_min) capsae[0].r_min = r;
        if (r > capsae[0].r_max) capsae[0].r_max = r;
        if (g < capsae[0].g_min) capsae[0].g_min = g;
        if (g > capsae[0].g_max) capsae[0].g_max = g;
        if (b < capsae[0].b_min) capsae[0].b_min = b;
        if (b > capsae[0].b_max) capsae[0].b_max = b;
        capsae[0].r_sum += r;
        capsae[0].g_sum += g;
        capsae[0].b_sum += b;
    }
    capsae[0].numerus = n_pix;

    /* scinde capsas donec n_colorum habeamus */
    while (n_capsarum < n_colorum) {
        /* invenire capsam cum maxima amplitudine */
        int optima = -1;
        int max_amp = 0;
        for (int i = 0; i < n_capsarum; i++) {
            if (capsae[i].numerus < 2) continue;
            int amp = capsa_magnitudo(&capsae[i]);
            if (amp > max_amp) { max_amp = amp; optima = i; }
        }
        if (optima < 0) break;

        /* scinde per medianum dimensionis maximae */
        int dim = capsa_amplitudo(&capsae[optima]);
        int medio;
        if (dim == 0) medio = (capsae[optima].r_min + capsae[optima].r_max) / 2;
        else if (dim == 1) medio = (capsae[optima].g_min + capsae[optima].g_max) / 2;
        else medio = (capsae[optima].b_min + capsae[optima].b_max) / 2;

        /* crea duas novas capsas ex pixelibus */
        capsa_t lo = {255, 0, 255, 0, 255, 0, 0, 0, 0, 0};
        capsa_t hi = {255, 0, 255, 0, 255, 0, 0, 0, 0, 0};

        for (int i = 0; i < n_pix; i++) {
            int r = rgb[i * 3 + 0];
            int g = rgb[i * 3 + 1];
            int b = rgb[i * 3 + 2];
            int v = (dim == 0) ? r : (dim == 1) ? g : b;

            /* est in capsa originali? */
            if (r < capsae[optima].r_min || r > capsae[optima].r_max) continue;
            if (g < capsae[optima].g_min || g > capsae[optima].g_max) continue;
            if (b < capsae[optima].b_min || b > capsae[optima].b_max) continue;

            capsa_t *dest = (v <= medio) ? &lo : &hi;
            if (r < dest->r_min) dest->r_min = r;
            if (r > dest->r_max) dest->r_max = r;
            if (g < dest->g_min) dest->g_min = g;
            if (g > dest->g_max) dest->g_max = g;
            if (b < dest->b_min) dest->b_min = b;
            if (b > dest->b_max) dest->b_max = b;
            dest->r_sum += r;
            dest->g_sum += g;
            dest->b_sum += b;
            dest->numerus++;
        }

        if (lo.numerus == 0 || hi.numerus == 0) {
            /* non potest scindere — signa ut non scindibilem */
            capsae[optima].r_min = capsae[optima].r_max;
            capsae[optima].g_min = capsae[optima].g_max;
            capsae[optima].b_min = capsae[optima].b_max;
            continue;
        }

        capsae[optima] = lo;
        capsae[n_capsarum] = hi;
        n_capsarum++;
    }

    /* mediam cuiusque capsae in paletam scribe */
    for (int i = 0; i < n_colorum; i++) {
        if (i < n_capsarum && capsae[i].numerus > 0) {
            paleta[i][0] = (uint8_t)(capsae[i].r_sum / capsae[i].numerus);
            paleta[i][1] = (uint8_t)(capsae[i].g_sum / capsae[i].numerus);
            paleta[i][2] = (uint8_t)(capsae[i].b_sum / capsae[i].numerus);
        } else {
            paleta[i][0] = paleta[i][1] = paleta[i][2] = 0;
        }
    }

    free(capsae);
}

/* ================================================================
 * Bayer dithering et indicem paletae invenire
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

static int indicem_proximum(const uint8_t paleta[][3], int n,
                             int r, int g, int b)
{
    int optimus = 0;
    int min_dist = 0x7FFFFFFF;
    for (int i = 0; i < n; i++) {
        int dr = r - paleta[i][0];
        int dg = g - paleta[i][1];
        int db = b - paleta[i][2];
        int dist = dr * dr + dg * dg + db * db;
        if (dist < min_dist) { min_dist = dist; optimus = i; }
    }
    return optimus;
}

/* ================================================================
 * LZW compressor
 * ================================================================ */

typedef struct {
    FILE *plica;
    int mag_codis;          /* current code size (bits) */
    int cod_purgandi;       /* clear code */
    int cod_finis;          /* EOI code */
    int prox_codex;         /* next available code */

    /* hash tabula pro dictionario */
    int16_t hash_praef[LZW_HASH_MAG];
    uint8_t hash_suff[LZW_HASH_MAG];
    int16_t hash_codex[LZW_HASH_MAG];

    /* bit packing (LSB first) */
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
    if (s->prox_codex >= LZW_MAX_CODEX) return;
    int h = ((praef << 8) ^ suff) % LZW_HASH_MAG;
    while (s->hash_codex[h] >= 0)
        h = (h + 1) % LZW_HASH_MAG;
    s->hash_praef[h] = (int16_t)praef;
    s->hash_suff[h]  = suff;
    s->hash_codex[h] = (int16_t)s->prox_codex;
    s->prox_codex++;

    /* code size crescit */
    if (s->prox_codex > (1 << s->mag_codis) && s->mag_codis < 12)
        s->mag_codis++;
}

static void lzw_comprime(FILE *plica, const uint8_t *data, int n)
{
    lzw_t s;
    memset(&s, 0, sizeof(s));
    s.plica = plica;
    s.cod_purgandi = 1 << LZW_MIN_MAG;
    s.cod_finis    = s.cod_purgandi + 1;
    memset(s.hash_codex, -1, sizeof(s.hash_codex));

    /* minimum code size */
    fputc(LZW_MIN_MAG, plica);

    lzw_dict_purga(&s);
    lzw_codicem_emitte(&s, s.cod_purgandi);

    int w = data[0]; /* current string code */
    for (int i = 1; i < n; i++) {
        uint8_t k = data[i];
        int wk = lzw_quaere(&s, w, k);
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

    /* flush residual bits */
    if (s.bit_numerus > 0)
        lzw_byte_emitte(&s, (uint8_t)(s.bit_alveus & 0xFF));
    lzw_sub_emitte(&s);

    /* block terminator */
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

pfr_gif_t *pfr_gif_initia(const char *via, int lat, int alt,
                            int mora_cs, int scala)
{
    if (scala < 1) scala = 1;

    pfr_gif_t *g = (pfr_gif_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->lat_fons = lat;
    g->alt_fons = alt;
    g->scala    = scala;
    g->lat      = lat / scala;
    g->alt      = alt / scala;
    g->mora_cs  = mora_cs;

    g->plica = fopen(via, "wb");
    if (!g->plica) { free(g); return NULL; }

    /* caput GIF89a — scribetur post paletam generatam */
    /* reserva spatium, revolvimus post primam tabulam */

    return g;
}

static void gif_caput_scribe(pfr_gif_t *g)
{
    FILE *f = g->plica;

    /* GIF89a */
    fwrite("GIF89a", 1, 6, f);
    scribe_u16le(f, (uint16_t)g->lat);
    scribe_u16le(f, (uint16_t)g->alt);

    /* packed: GCT=1, color_res=7, sort=0, gct_size=PALETA_POTENTIA-1 */
    fputc(0x80 | (6 << 4) | (PALETA_POTENTIA - 1), f);
    fputc(0, f);    /* background color */
    fputc(0, f);    /* pixel aspect ratio */

    /* Global Color Table — 2^PALETA_POTENTIA entries */
    int n_entries = 1 << PALETA_POTENTIA;
    for (int i = 0; i < n_entries; i++) {
        if (i < PALETA_MAG) {
            fputc(g->paleta[i][0], f);
            fputc(g->paleta[i][1], f);
            fputc(g->paleta[i][2], f);
        } else {
            fputc(0, f); fputc(0, f); fputc(0, f);
        }
    }

    /* Netscape Application Extension — circulus infinitus */
    fputc(0x21, f);     /* extension introducer */
    fputc(0xFF, f);     /* application extension */
    fputc(11, f);       /* block size */
    fwrite("NETSCAPE2.0", 1, 11, f);
    fputc(3, f);        /* sub-block size */
    fputc(1, f);        /* sub-block id */
    scribe_u16le(f, 0); /* loop count: 0 = infinitus */
    fputc(0, f);        /* terminator */
}

int pfr_gif_tabulam_adde(pfr_gif_t *g, const uint32_t *pixels)
{
    if (!g || !pixels) return -1;

    int n_pix = g->lat * g->alt;

    /* scala et converte ARGB -> RGB */
    uint8_t *rgb = (uint8_t *)malloc((size_t)n_pix * 3);
    if (!rgb) return -1;

    for (int y = 0; y < g->alt; y++) {
        for (int x = 0; x < g->lat; x++) {
            /* media pixelorum in scala x scala area */
            int r_sum = 0, g_sum = 0, b_sum = 0;
            int cnt = 0;
            for (int sy = 0; sy < g->scala; sy++) {
                int fy = y * g->scala + sy;
                if (fy >= g->alt_fons) continue;
                for (int sx = 0; sx < g->scala; sx++) {
                    int fx = x * g->scala + sx;
                    if (fx >= g->lat_fons) continue;
                    uint32_t px = pixels[fy * g->lat_fons + fx];
                    r_sum += (px >> 16) & 0xFF;
                    g_sum += (px >> 8)  & 0xFF;
                    b_sum +=  px        & 0xFF;
                    cnt++;
                }
            }
            if (cnt > 0) {
                int idx = (y * g->lat + x) * 3;
                rgb[idx + 0] = (uint8_t)(r_sum / cnt);
                rgb[idx + 1] = (uint8_t)(g_sum / cnt);
                rgb[idx + 2] = (uint8_t)(b_sum / cnt);
            }
        }
    }

    /* genera paletam ex prima tabula */
    if (!g->paleta_parata) {
        paletam_genera(rgb, n_pix, g->paleta, PALETA_MAG);
        g->paleta_parata = 1;
        gif_caput_scribe(g);
    }

    /* converte in indices paletae cum Bayer dithering */
    uint8_t *indices = (uint8_t *)malloc((size_t)n_pix);
    if (!indices) { free(rgb); return -1; }

    for (int y = 0; y < g->alt; y++) {
        for (int x = 0; x < g->lat; x++) {
            int idx = (y * g->lat + x) * 3;
            double threshold = (bayer_8x8[y & 7][x & 7] / 64.0) - 0.5;
            double spread = 24.0;
            int r = (int)(rgb[idx + 0] + threshold * spread);
            int gv = (int)(rgb[idx + 1] + threshold * spread);
            int b = (int)(rgb[idx + 2] + threshold * spread);
            if (r < 0) r = 0; if (r > 255) r = 255;
            if (gv < 0) gv = 0; if (gv > 255) gv = 255;
            if (b < 0) b = 0; if (b > 255) b = 255;
            indices[y * g->lat + x] = (uint8_t)indicem_proximum(
                g->paleta, PALETA_MAG, r, gv, b);
        }
    }

    free(rgb);

    FILE *f = g->plica;

    /* Graphic Control Extension */
    fputc(0x21, f);
    fputc(0xF9, f);
    fputc(4, f);
    fputc(0x08, f);     /* disposal = 2 (restore to bg), no transparency */
    scribe_u16le(f, (uint16_t)g->mora_cs);
    fputc(0, f);        /* transparent color index */
    fputc(0, f);        /* terminator */

    /* Image Descriptor */
    fputc(0x2C, f);
    scribe_u16le(f, 0);                    /* left */
    scribe_u16le(f, 0);                    /* top */
    scribe_u16le(f, (uint16_t)g->lat);     /* width */
    scribe_u16le(f, (uint16_t)g->alt);     /* height */
    fputc(0, f);                            /* packed: no LCT */

    /* LZW compressed data */
    lzw_comprime(f, indices, n_pix);

    free(indices);
    g->numerus++;
    return 0;
}

void pfr_gif_fini(pfr_gif_t *g)
{
    if (!g) return;
    if (g->plica) {
        fputc(0x3B, g->plica);     /* GIF trailer */
        fclose(g->plica);
    }
    free(g);
}
