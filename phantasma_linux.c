/*
 * phantasma_linux.c — implementatio X11 (Linux)
 * ================================================
 *
 * Fenestram per Xlib creat, alveum pixelorum per XImage reddit.
 * Eventus X11 in PFR eventus convertit.
 */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "phantasma.h"

/* ================================================================
 * status globalis
 * ================================================================ */

#define CODA_MAX 256

static struct {
    int initiatum;
    uint64_t tempus_initii_ns;
    char erratum[256];

    /* coda eventuum */
    pfr_eventus_t eventus[CODA_MAX];
    int caput;
    int cauda;

    /* status clavium */
    pfr_u8 claves[PFR_SC_NUMERUS];

    /* positio muris */
    int muris_x;
    int muris_y;

    /* X11 */
    Display *exhibitio;
    Atom wm_dele;
} ph;

static uint64_t tempus_nunc_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ================================================================
 * structurae internae
 * ================================================================ */

struct pfr_fenestra {
    int latitudo;
    int altitudo;
    Window x_fenestra;
    GC x_gc;
};

struct pfr_pictor {
    pfr_fenestra_t *fenestra;
    int latitudo;
    int altitudo;
    uint32_t *alveus;
    pfr_u8 color_r, color_g, color_b, color_a;
};

struct pfr_textura {
    int latitudo;
    int altitudo;
    uint32_t *pixels;
};

/* ================================================================
 * hook redimensionationis — vocatur ex communia.c
 * ================================================================ */

static void pictorem_post_redimensiona(pfr_pictor_t *p)
{
    (void)p;
}

/* ================================================================
 * functiones communes (coda, textura, pictor, claves, pausa)
 * ================================================================ */

#include "communia.c"

/* ================================================================
 * tabula clavium: X11 KeySym -> PFR scancode
 * ================================================================ */

static int x11_ad_scancodem(KeySym ks)
{
    if (ks >= XK_a && ks <= XK_z)
        return PFR_SC_A + (int)(ks - XK_a);
    if (ks >= XK_1 && ks <= XK_9)
        return PFR_SC_1 + (int)(ks - XK_1);
    if (ks == XK_0)
        return PFR_SC_0;
    switch (ks) {
    case XK_Left:    return PFR_SC_SINISTRUM;
    case XK_Right:   return PFR_SC_DEXTRUM;
    case XK_Up:      return PFR_SC_SURSUM;
    case XK_Down:    return PFR_SC_DEORSUM;
    case XK_space:   return PFR_SC_SPATIUM;
    case XK_Tab:     return PFR_SC_TABULA;
    case XK_Escape:  return PFR_SC_EFFUGIUM;
    case XK_minus:   return PFR_SC_MINUS;
    case XK_equal:   return PFR_SC_AEQUALE;
    default:         return 0;
    }
}

static int x11_ad_symbolum(KeySym ks)
{
    /* ASCII directe */
    if (ks >= XK_space && ks <= XK_asciitilde)
        return (int)ks;
    switch (ks) {
    case XK_Escape:  return PFR_CL_EFFUGIUM;
    case XK_Tab:     return PFR_CL_TABULA;
    case XK_Left:    return PFR_CL_SINISTRUM;
    case XK_Right:   return PFR_CL_DEXTRUM;
    case XK_Up:      return PFR_CL_SURSUM;
    case XK_Down:    return PFR_CL_DEORSUM;
    default:         return 0;
    }
}

/* ================================================================
 * pfr_initia / pfr_fini
 * ================================================================ */

int pfr_initia(pfr_u32 flags)
{
    (void)flags;
    memset(&ph, 0, sizeof(ph));

    ph.tempus_initii_ns = tempus_nunc_ns();

    ph.exhibitio = XOpenDisplay(NULL);
    if (!ph.exhibitio) {
        snprintf(
            ph.erratum, sizeof(ph.erratum),
            "XOpenDisplay fallivit"
        );
        return -1;
    }

    ph.wm_dele   = XInternAtom(ph.exhibitio, "WM_DELETE_WINDOW", False);
    ph.initiatum = 1;
    return 0;
}

void pfr_fini(void)
{
    if (ph.exhibitio) {
        XCloseDisplay(ph.exhibitio);
        ph.exhibitio = NULL;
    }
    ph.initiatum = 0;
}

/* ================================================================
 * fenestra
 * ================================================================ */

pfr_fenestra_t *pfr_fenestram_crea(
    const char *titulus, int x, int y,
    int lat, int alt, pfr_u32 flags
) {
    (void)x;
    (void)y;
    (void)flags;
    if (!ph.exhibitio)
        return NULL;

    pfr_fenestra_t *f = (pfr_fenestra_t *)calloc(1, sizeof(*f));
    if (!f)
        return NULL;
    f->latitudo = lat;
    f->altitudo = alt;

    int scr = DefaultScreen(ph.exhibitio);
    f->x_fenestra = XCreateSimpleWindow(
        ph.exhibitio, RootWindow(ph.exhibitio, scr),
        0, 0, (unsigned)lat, (unsigned)alt, 0,
        BlackPixel(ph.exhibitio, scr),
        BlackPixel(ph.exhibitio, scr)
    );

    XStoreName(ph.exhibitio, f->x_fenestra, titulus);
    XSelectInput(
        ph.exhibitio, f->x_fenestra,
        ExposureMask | KeyPressMask | KeyReleaseMask
        | ButtonPressMask | PointerMotionMask | StructureNotifyMask
    );

    XSetWMProtocols(ph.exhibitio, f->x_fenestra, &ph.wm_dele, 1);

    f->x_gc = XCreateGC(ph.exhibitio, f->x_fenestra, 0, NULL);

    XMapWindow(ph.exhibitio, f->x_fenestra);
    XFlush(ph.exhibitio);

    /* exspecta MapNotify */
    XEvent xe;
    while (1) {
        XNextEvent(ph.exhibitio, &xe);
        if (xe.type == MapNotify)
            break;
    }

    return f;
}

void pfr_fenestram_destrue(pfr_fenestra_t *f)
{
    if (!f || !ph.exhibitio)
        return;
    XFreeGC(ph.exhibitio, f->x_gc);
    XDestroyWindow(ph.exhibitio, f->x_fenestra);
    XFlush(ph.exhibitio);
    free(f);
}

/* ================================================================
 * pictor (renderer)
 * ================================================================ */

pfr_pictor_t *pfr_pictorem_crea(
    pfr_fenestra_t *f, int index,
    pfr_u32 flags
) {
    (void)index;
    (void)flags;
    if (!f)
        return NULL;

    pfr_pictor_t *p = (pfr_pictor_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->fenestra = f;
    p->latitudo = f->latitudo;
    p->altitudo = f->altitudo;
    p->alveus = (uint32_t *)calloc(
        (size_t)p->latitudo * p->altitudo,
        sizeof(uint32_t)
    );
    if (!p->alveus) {
        free(p);
        return NULL;
    }
    return p;
}

void pfr_pictorem_destrue(pfr_pictor_t *p)
{
    if (!p)
        return;
    free(p->alveus);
    free(p);
}

/* ================================================================
 * redditio fenestrae
 * ================================================================ */

void pfr_praesenta(pfr_pictor_t *p)
{
    if (!p || !p->fenestra || !ph.exhibitio)
        return;

    int scr     = DefaultScreen(ph.exhibitio);
    Visual *vis = DefaultVisual(ph.exhibitio, scr);
    int depth   = DefaultDepth(ph.exhibitio, scr);

    XImage *img = XCreateImage(
        ph.exhibitio, vis, (unsigned)depth, ZPixmap, 0,
        (char *)p->alveus,
        (unsigned)p->latitudo, (unsigned)p->altitudo,
        32, p->latitudo * 4
    );
    if (!img)
        return;

    /* ne XDestroyImage data liberet */
    img->f.destroy_image = NULL;

    XPutImage(
        ph.exhibitio, p->fenestra->x_fenestra,
        p->fenestra->x_gc, img,
        0, 0, 0, 0,
        (unsigned)p->latitudo, (unsigned)p->altitudo
    );

    /* manu liberamus sine data */
    img->data = NULL;
    XDestroyImage(img);

    XFlush(ph.exhibitio);
}

/* ================================================================
 * eventus
 * ================================================================ */

static void x11_eventum_convertere(XEvent *xe)
{
    if (xe->type == KeyPress || xe->type == KeyRelease) {
        /* symbolum cum modifiers */
        char alveus[4] = {0};
        KeySym ks;
        XLookupString(&xe->xkey, alveus, sizeof(alveus), &ks, NULL);

        /* scancode ex keysym sine modifiers */
        KeySym ks_base = XLookupKeysym(&xe->xkey, 0);
        int scancode   = x11_ad_scancodem(ks_base);
        int symbolum   = x11_ad_symbolum(ks);
        /* characteres moderatores ex XLookupString praeferre */
        if (alveus[0] > 0 && (alveus[0] < 0x20 || alveus[0] == 0x7f))
            symbolum = (int)(unsigned char)alveus[0];
        int depressus  = (xe->type == KeyPress) ? 1 : 0;

        if (scancode > 0 && scancode < PFR_SC_NUMERUS)
            ph.claves[scancode] = depressus ? 1 : 0;

        pfr_eventus_t se;
        memset(&se, 0, sizeof(se));
        se.typus = depressus ? PFR_CLAVIS_INF : PFR_CLAVIS_SUR;
        se.clavis.typus     = se.typus;
        se.clavis.tempus    = pfr_tempus();
        se.clavis.status    = depressus;
        se.clavis.signum.scancodex = scancode;
        se.clavis.signum.symbolum  = symbolum;
        coda_insere(&se);
    }

    /* muris */
    if (xe->type == ButtonPress) {
        /* rota: Button4 = sursum, Button5 = deorsum */
        if (xe->xbutton.button == 4 || xe->xbutton.button == 5) {
            pfr_eventus_t se;
            memset(&se, 0, sizeof(se));
            se.typus      = PFR_ROTA_MURIS;
            se.rota.typus = PFR_ROTA_MURIS;
            se.rota.y     = (xe->xbutton.button == 4) ? 1 : -1;
            coda_insere(&se);
        }
        /* plectra 1-3 */
        if (xe->xbutton.button >= 1 && xe->xbutton.button <= 3) {
            int mx     = xe->xbutton.x;
            int my     = xe->xbutton.y;
            ph.muris_x = mx;
            ph.muris_y = my;
            pfr_eventus_t se;
            memset(&se, 0, sizeof(se));
            se.typus          = PFR_MURIS_INF;
            se.muris.typus    = PFR_MURIS_INF;
            se.muris.tempus   = pfr_tempus();
            se.muris.x        = mx;
            se.muris.y        = my;
            se.muris.plectrum = (int)xe->xbutton.button;
            coda_insere(&se);
        }
    }

    /* motus muris */
    if (xe->type == MotionNotify) {
        ph.muris_x = xe->xmotion.x;
        ph.muris_y = xe->xmotion.y;
    }

    /* fenestra redimensionata */
    if (xe->type == ConfigureNotify) {
        int lat = xe->xconfigure.width;
        int alt = xe->xconfigure.height;
        pfr_eventus_t se;
        memset(&se, 0, sizeof(se));
        se.typus             = PFR_FENESTRA_MUTATA;
        se.fenestra.typus    = PFR_FENESTRA_MUTATA;
        se.fenestra.tempus   = pfr_tempus();
        se.fenestra.lat      = lat;
        se.fenestra.alt      = alt;
        coda_insere(&se);
    }

    /* fenestra clausa */
    if (
        xe->type == ClientMessage
        && (Atom)xe->xclient.data.l[0] == ph.wm_dele
    ) {
        pfr_eventus_t se;
        memset(&se, 0, sizeof(se));
        se.typus = PFR_EXITUS;
        coda_insere(&se);
    }
}

int pfr_eventum_lege(pfr_eventus_t *eventus)
{
    if (!ph.exhibitio)
        return 0;

    while (XPending(ph.exhibitio) > 0) {
        XEvent xe;
        XNextEvent(ph.exhibitio, &xe);
        x11_eventum_convertere(&xe);
    }

    if (eventus)
        return coda_extrahe(eventus);
    return (ph.cauda != ph.caput) ? 1 : 0;
}

/* ================================================================
 * tempus
 * ================================================================ */

pfr_u32 pfr_tempus(void)
{
    uint64_t delta = tempus_nunc_ns() - ph.tempus_initii_ns;
    return (pfr_u32)(delta / 1000000ULL);
}
