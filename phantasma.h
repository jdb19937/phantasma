/*
 * phantasma.h — PFR: Phantasma Fenestrae Rudimentalis
 * =====================================================
 *
 * Bibliotheca fenestrae minimalis, loco libSDL2.
 * Mac (Cocoa) et Linux (X11) sustinet sine dependentiis externis.
 *
 * Solae functiones quae exemplis sufficiunt implementantur:
 *   fenestra, pictor, textura, eventus, claves, tempus.
 */

#ifndef PHANTASMA_H
#define PHANTASMA_H

#include <stdint.h>

/* ================================================================
 * typi fundamentales
 * ================================================================ */

typedef uint32_t pfr_u32;
typedef uint8_t  pfr_u8;

/* ================================================================
 * constantiae initiationis
 * ================================================================ */

#define PFR_INITIA_VIDEO    0x01u

/* ================================================================
 * positio fenestrae
 * ================================================================ */

#define PFR_POS_MEDIUM      0x7FFFu

/* ================================================================
 * flags pictoris
 * ================================================================ */

#define PFR_PICTOR_CELER    0x01u   /* acceleratus */
#define PFR_PICTOR_SYNC     0x02u   /* vsync */

/* ================================================================
 * forma pixelorum et accessus texturae
 * ================================================================ */

#define PFR_PIXEL_ARGB8888  1u
#define PFR_TEXTURA_FLUENS  1       /* streaming */

/* ================================================================
 * typi eventuum
 * ================================================================ */

#define PFR_EXITUS          0x01    /* fenestra clausa */
#define PFR_CLAVIS_INF      0x02    /* clavis pressa (infra) */
#define PFR_CLAVIS_SUR      0x03    /* clavis missa (sursum) */
#define PFR_ROTA_MURIS      0x04    /* rota muris */

/* ================================================================
 * symbola clavium — ASCII ubi possibile
 * ================================================================ */

#define PFR_CL_TABULA       '\t'
#define PFR_CL_EFFUGIUM     '\x1b'
#define PFR_CL_SPATIUM      ' '
#define PFR_CL_PLUS         '+'
#define PFR_CL_MINUS        '-'
#define PFR_CL_AEQUALE      '='
#define PFR_CL_0            '0'
#define PFR_CL_1            '1'
#define PFR_CL_2            '2'
#define PFR_CL_3            '3'
#define PFR_CL_4            '4'
#define PFR_CL_5            '5'
#define PFR_CL_6            '6'
#define PFR_CL_7            '7'
#define PFR_CL_8            '8'
#define PFR_CL_9            '9'

/* litterae — usa characterem directe: 'a', 'q', 'w', etc. */

/* claves non-ASCII */
#define PFR_CL_DEXTRUM      0x40000001
#define PFR_CL_SINISTRUM    0x40000002
#define PFR_CL_DEORSUM      0x40000003
#define PFR_CL_SURSUM       0x40000004

/* ================================================================
 * scancodes (USB HID)
 * ================================================================ */

enum {
    PFR_SC_A        = 4,
    PFR_SC_B        = 5,
    PFR_SC_C        = 6,
    PFR_SC_D        = 7,
    PFR_SC_E        = 8,
    PFR_SC_F        = 9,
    PFR_SC_G        = 10,
    PFR_SC_H        = 11,
    PFR_SC_I        = 12,
    PFR_SC_J        = 13,
    PFR_SC_K        = 14,
    PFR_SC_L        = 15,
    PFR_SC_M        = 16,
    PFR_SC_N        = 17,
    PFR_SC_O        = 18,
    PFR_SC_P        = 19,
    PFR_SC_Q        = 20,
    PFR_SC_R        = 21,
    PFR_SC_S        = 22,
    PFR_SC_T        = 23,
    PFR_SC_U        = 24,
    PFR_SC_V        = 25,
    PFR_SC_W        = 26,
    PFR_SC_X        = 27,
    PFR_SC_Y        = 28,
    PFR_SC_Z        = 29,
    PFR_SC_1        = 30,
    PFR_SC_2        = 31,
    PFR_SC_3        = 32,
    PFR_SC_4        = 33,
    PFR_SC_5        = 34,
    PFR_SC_6        = 35,
    PFR_SC_7        = 36,
    PFR_SC_8        = 37,
    PFR_SC_9        = 38,
    PFR_SC_0        = 39,
    PFR_SC_EFFUGIUM = 41,
    PFR_SC_TABULA   = 43,
    PFR_SC_SPATIUM  = 44,
    PFR_SC_MINUS    = 45,
    PFR_SC_AEQUALE  = 46,
    PFR_SC_DEXTRUM  = 79,
    PFR_SC_SINISTRUM = 80,
    PFR_SC_DEORSUM  = 81,
    PFR_SC_SURSUM   = 82,
    PFR_SC_NUMERUS  = 512
};

/* ================================================================
 * typi opaci
 * ================================================================ */

typedef struct pfr_fenestra pfr_fenestra_t;
typedef struct pfr_pictor   pfr_pictor_t;
typedef struct pfr_textura  pfr_textura_t;

/* ================================================================
 * pfr_rectum_t
 * ================================================================ */

typedef struct pfr_rectum {
    int x, y, lat, alt;    /* lat = latitudo, alt = altitudo */
} pfr_rectum_t;

/* ================================================================
 * pfr_eventus_t
 * ================================================================ */

typedef struct pfr_clavis_signum {
    int scancodex;
    int symbolum;
    int moduli;
} pfr_clavis_signum_t;

typedef struct pfr_eventus_clavis {
    pfr_u32 typus;
    pfr_u32 tempus;
    pfr_u8  status;
    pfr_u8  repetitio;
    pfr_u8  repletus[2];
    pfr_clavis_signum_t signum;
} pfr_eventus_clavis_t;

typedef struct pfr_eventus_rota {
    pfr_u32 typus;
    pfr_u32 tempus;
    int x;
    int y;
} pfr_eventus_rota_t;

typedef union pfr_eventus {
    pfr_u32                typus;
    pfr_eventus_clavis_t   clavis;
    pfr_eventus_rota_t     rota;
} pfr_eventus_t;

/* ================================================================
 * functiones
 * ================================================================ */

/* --- initium et finis --- */

int          pfr_initia(pfr_u32 flags);
void         pfr_fini(void);
const char  *pfr_erratum(void);

/* --- fenestra --- */

pfr_fenestra_t *pfr_fenestram_crea(
    const char *titulus, int x, int y,
    int lat, int alt, pfr_u32 flags
);
void            pfr_fenestram_destrue(pfr_fenestra_t *f);

/* --- pictor (renderer) --- */

pfr_pictor_t *pfr_pictorem_crea(
    pfr_fenestra_t *f, int index,
    pfr_u32 flags
);
void          pfr_pictorem_destrue(pfr_pictor_t *p);

/* --- textura --- */

pfr_textura_t *pfr_texturam_crea(
    pfr_pictor_t *p, pfr_u32 forma,
    int accessus, int lat, int alt
);
void           pfr_texturam_destrue(pfr_textura_t *t);

int  pfr_texturam_renova(
    pfr_textura_t *t, const pfr_rectum_t *rect,
    const void *pixels, int passus
);

/* --- redditio --- */

int  pfr_purga(pfr_pictor_t *p);
int  pfr_texturam_pinge(
    pfr_pictor_t *p, pfr_textura_t *t,
    const pfr_rectum_t *fons, const pfr_rectum_t *dest
);
void pfr_praesenta(pfr_pictor_t *p);

int  pfr_colorem_pone(
    pfr_pictor_t *p,
    pfr_u8 r, pfr_u8 g, pfr_u8 b, pfr_u8 a
);
int  pfr_punctum_pinge(pfr_pictor_t *p, int x, int y);
int  pfr_rectum_imple(pfr_pictor_t *p, const pfr_rectum_t *rect);

/* --- eventus --- */

int             pfr_eventum_lege(pfr_eventus_t *e);
const pfr_u8   *pfr_claves_status(int *numerus);

/* --- tempus --- */

pfr_u32  pfr_tempus(void);
void     pfr_pausa(pfr_u32 ms);

/* ================================================================
 * inscriptores — GIF et MP4
 * ================================================================ */

typedef struct pfr_gif pfr_gif_t;
typedef struct pfr_mp4 pfr_mp4_t;

/*
 * pfr_gif_initia — inscriptorem GIF animatum creat.
 * mora_cs: mora inter tabulas in centisecundis (e.g. 3 = ~33fps).
 * scala:   factor reductionis (1 = plena, 2 = dimidia, etc.).
 * pixels ARGB8888 expectantur.
 */
pfr_gif_t *pfr_gif_initia(
    const char *via, int lat, int alt,
    int mora_cs, int scala
);
int  pfr_gif_tabulam_adde(pfr_gif_t *g, const uint32_t *pixels);
void pfr_gif_fini(pfr_gif_t *g);

/*
 * pfr_mp4_initia — inscriptorem MP4 (H.264 Baseline) creat.
 * fps: tabulae per secundam (e.g. 30).
 * pixels ARGB8888 expectantur.
 */
pfr_mp4_t *pfr_mp4_initia(const char *via, int lat, int alt, int fps);
int  pfr_mp4_tabulam_adde(pfr_mp4_t *m, const uint32_t *pixels);
void pfr_mp4_fini(pfr_mp4_t *m);

#endif /* PHANTASMA_H */
