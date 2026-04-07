/*
 * pfr_communia.c — functiones communes inter platformas
 * =======================================================
 *
 * Coda eventuum, textura, pictor operationes, color, pausa.
 * Ab utraque implementatione (Darwin et Linux) includitur.
 */

/* ================================================================
 * coda eventuum
 * ================================================================ */

static void coda_insere(const pfr_eventus_t *e)
{
    int prox = (ph.caput + 1) % CODA_MAX;
    if (prox == ph.cauda)
        return;
    ph.eventus[ph.caput] = *e;
    ph.caput = prox;
}

static int coda_extrahe(pfr_eventus_t *e)
{
    if (ph.cauda == ph.caput)
        return 0;
    *e       = ph.eventus[ph.cauda];
    ph.cauda = (ph.cauda + 1) % CODA_MAX;
    return 1;
}

/* ================================================================
 * erratum
 * ================================================================ */

const char *pfr_erratum(void)
{
    return ph.erratum;
}

/* ================================================================
 * textura
 * ================================================================ */

pfr_textura_t *pfr_texturam_crea(
    pfr_pictor_t *p, pfr_u32 forma,
    int accessus, int lat, int alt
) {
    (void)p;
    (void)forma;
    (void)accessus;
    pfr_textura_t *t = (pfr_textura_t *)calloc(1, sizeof(*t));
    if (!t)
        return NULL;
    t->latitudo = lat;
    t->altitudo = alt;
    t->pixels   = (uint32_t *)calloc((size_t)lat * alt, sizeof(uint32_t));
    if (!t->pixels) {
        free(t);
        return NULL;
    }
    return t;
}

void pfr_texturam_destrue(pfr_textura_t *t)
{
    if (!t)
        return;
    free(t->pixels);
    free(t);
}

int pfr_texturam_renova(
    pfr_textura_t *t, const pfr_rectum_t *rect,
    const void *pixels, int passus
) {
    if (!t || !pixels)
        return -1;
    if (rect) {
        const unsigned char *fons = (const unsigned char *)pixels;
        for (int y = 0; y < rect->alt; y++)
            memcpy(
                t->pixels + (rect->y + y) * t->latitudo + rect->x,
                fons + y * passus,
                (size_t)rect->lat * 4
            );
    } else {
        if (passus == t->latitudo * 4)
            memcpy(
                t->pixels, pixels,
                (size_t)t->latitudo * t->altitudo * 4
            );
        else {
            const unsigned char *fons = (const unsigned char *)pixels;
            for (int y = 0; y < t->altitudo; y++)
                memcpy(
                    t->pixels + y * t->latitudo,
                    fons + y * passus,
                    (size_t)t->latitudo * 4
                );
        }
    }
    return 0;
}

/* ================================================================
 * operationes pictoris
 * ================================================================ */

int pfr_purga(pfr_pictor_t *p)
{
    if (!p)
        return -1;
    uint32_t color = (
        ((uint32_t)p->color_a << 24) |
        ((uint32_t)p->color_r << 16) |
        ((uint32_t)p->color_g << 8) |
        (uint32_t)p->color_b
    );
    size_t n = (size_t)p->latitudo * p->altitudo;
    if (color == 0)
        memset(p->alveus, 0, n * 4);
    else
        for (size_t i = 0; i < n; i++)
            p->alveus[i] = color;
    return 0;
}

int pfr_texturam_pinge(
    pfr_pictor_t *p, pfr_textura_t *t,
    const pfr_rectum_t *fons, const pfr_rectum_t *dest
) {
    if (!p || !t)
        return -1;
    (void)fons;
    (void)dest;
    size_t n = (size_t)p->latitudo * p->altitudo;
    memcpy(p->alveus, t->pixels, n * sizeof(uint32_t));
    return 0;
}

int pfr_colorem_pone(
    pfr_pictor_t *p,
    pfr_u8 r, pfr_u8 g, pfr_u8 b, pfr_u8 a
) {
    if (!p)
        return -1;
    p->color_r = r;
    p->color_g = g;
    p->color_b = b;
    p->color_a = a;
    return 0;
}

int pfr_punctum_pinge(pfr_pictor_t *p, int x, int y)
{
    if (!p)
        return -1;
    if (x < 0 || x >= p->latitudo || y < 0 || y >= p->altitudo)
        return 0;
    p->alveus[y * p->latitudo + x] = (
        ((uint32_t)p->color_a << 24) |
        ((uint32_t)p->color_r << 16) |
        ((uint32_t)p->color_g << 8) |
        (uint32_t)p->color_b
    );
    return 0;
}

int pfr_rectum_imple(pfr_pictor_t *p, const pfr_rectum_t *rect)
{
    if (!p || !rect)
        return -1;
    uint32_t color = (
        ((uint32_t)p->color_a << 24) |
        ((uint32_t)p->color_r << 16) |
        ((uint32_t)p->color_g << 8) |
        (uint32_t)p->color_b
    );
    int x0 = rect->x < 0 ? 0 : rect->x;
    int y0 = rect->y < 0 ? 0 : rect->y;
    int x1 = rect->x + rect->lat;
    int y1 = rect->y + rect->alt;
    if (x1 > p->latitudo)
        x1 = p->latitudo;
    if (y1 > p->altitudo)
        y1 = p->altitudo;
    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
            p->alveus[y * p->latitudo + x] = color;
    return 0;
}

/* ================================================================
 * claves
 * ================================================================ */

const pfr_u8 *pfr_claves_status(int *numerus)
{
    if (numerus)
        *numerus = PFR_SC_NUMERUS;
    return ph.claves;
}

/* ================================================================
 * muris — positio cursor
 * ================================================================ */

void pfr_muris_positio(int *x, int *y)
{
    if (x)
        *x = ph.muris_x;
    if (y)
        *y = ph.muris_y;
}

/* ================================================================
 * pausa
 * ================================================================ */

void pfr_pausa(pfr_u32 ms)
{
    struct timespec req;
    req.tv_sec  = ms / 1000;
    req.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&req, NULL);
}
