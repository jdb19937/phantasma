/*
 * phantasma_darwin.m — implementatio Cocoa (macOS)
 * ==================================================
 *
 * Fenestram per NSWindow/NSView creat, alveum pixelorum
 * per CoreGraphics reddit.  Eventus Cocoa in PFR eventus convertit.
 */

#import <Cocoa/Cocoa.h>
#include <mach/mach_time.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "phantasma.h"

/* ================================================================
 * status globalis
 * ================================================================ */

#define CODA_MAX 256

static struct {
    int initiatum;
    uint64_t tempus_initii;
    mach_timebase_info_data_t tempus_info;
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
} ph;

/* forward declarations pro classibus Cocoa */
@class PhantasmaView;
@class PhantasmaDelegate;

/* ================================================================
 * structurae internae
 * ================================================================ */

struct pfr_fenestra {
    int latitudo;
    int altitudo;
    NSWindow          *ns_fenestra;
    PhantasmaView     *ns_visus;
    PhantasmaDelegate *ns_delegatus;
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
 * functiones communes (coda, textura, pictor, claves, pausa)
 * ================================================================ */

#include "communia.c"

/* ================================================================
 * tabula clavium: macOS keycode -> PFR scancode
 * ================================================================ */

static const int mac_scancodex[128] = {
    [0x00] =  4, /* A */   [0x01] = 22, /* S */   [0x02] =  7, /* D */
    [0x03] =  9, /* F */   [0x04] = 11, /* H */   [0x05] = 10, /* G */
    [0x06] = 29, /* Z */   [0x07] = 27, /* X */   [0x08] =  6, /* C */
    [0x09] = 25, /* V */   [0x0B] =  5, /* B */   [0x0C] = 20, /* Q */
    [0x0D] = 26, /* W */   [0x0E] =  8, /* E */   [0x0F] = 21, /* R */
    [0x10] = 28, /* Y */   [0x11] = 23, /* T */   [0x12] = 30, /* 1 */
    [0x13] = 31, /* 2 */   [0x14] = 32, /* 3 */   [0x15] = 33, /* 4 */
    [0x16] = 34, /* 6 */   [0x17] = 35, /* 5 */   [0x18] = 46, /* = */
    [0x19] = 36, /* 9 */   [0x1A] = 37, /* 7 */   [0x1B] = 45, /* - */
    [0x1C] = 38, /* 8 */   [0x1D] = 39, /* 0 */   [0x1F] = 18, /* O */
    [0x20] = 24, /* U */   [0x22] = 12, /* I */   [0x23] = 19, /* P */
    [0x25] = 15, /* L */   [0x26] = 13, /* J */   [0x28] = 14, /* K */
    [0x2D] = 17, /* N */   [0x2E] = 16, /* M */   [0x30] = 43, /* Tab */
    [0x31] = 44, /* Space */   [0x35] = 41, /* Escape */
    [0x7B] = 80, /* Left */    [0x7C] = 79, /* Right */
    [0x7D] = 81, /* Down */    [0x7E] = 82, /* Up */
};

/* macOS NSEvent -> PFR symbolum clavis */
static int mac_ad_symbolum(NSEvent *e)
{
    NSString *chars = [e characters];
    if (chars && [chars length] > 0) {
        unichar ch = [chars characterAtIndex:0];
        if (ch == 0x1B) return PFR_CL_EFFUGIUM;
        if (ch == 0x09) return PFR_CL_TABULA;
        if (ch >= 0x20 && ch < 0x7F) return (int)ch;
        switch (ch) {
        case NSUpArrowFunctionKey:    return PFR_CL_SURSUM;
        case NSDownArrowFunctionKey:  return PFR_CL_DEORSUM;
        case NSLeftArrowFunctionKey:  return PFR_CL_SINISTRUM;
        case NSRightArrowFunctionKey: return PFR_CL_DEXTRUM;
        }
    }
    switch ([e keyCode]) {
    case 0x35: return PFR_CL_EFFUGIUM;
    case 0x30: return PFR_CL_TABULA;
    case 0x31: return PFR_CL_SPATIUM;
    default:   return 0;
    }
}

/* ================================================================
 * PhantasmaView — NSView quae alveum pixelorum reddit
 * ================================================================ */

@interface PhantasmaView : NSView
{
@public
    uint32_t *alveus;
    int lat;
    int alt;
}
@end

@implementation PhantasmaView

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)canBecomeKeyView      { return YES; }
- (void)keyDown:(NSEvent *)e   { (void)e; }
- (void)keyUp:(NSEvent *)e     { (void)e; }

- (void)drawRect:(NSRect)dirtyRect
{
    (void)dirtyRect;
    if (!alveus) return;

    CGContextRef ctx = [[NSGraphicsContext currentContext] CGContext];

    /* flip: y=0 ad summum */
    CGContextTranslateCTM(ctx, 0, self.bounds.size.height);
    CGContextScaleCTM(ctx, 1, -1);

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef bctx = CGBitmapContextCreate(
        alveus, (size_t)lat, (size_t)alt, 8, (size_t)lat * 4, cs,
        kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
    CGImageRef img = CGBitmapContextCreateImage(bctx);
    CGContextDrawImage(ctx, CGRectMake(0, 0, lat, alt), img);
    CGImageRelease(img);
    CGContextRelease(bctx);
    CGColorSpaceRelease(cs);
}

- (void)scrollWheel:(NSEvent *)e
{
    double dy = [e scrollingDeltaY];
    if ([e hasPreciseScrollingDeltas])
        dy /= 10.0;
    if (dy > 0.1 || dy < -0.1) {
        pfr_eventus_t se;
        memset(&se, 0, sizeof(se));
        se.typus = PFR_ROTA_MURIS;
        se.rota.typus = PFR_ROTA_MURIS;
        se.rota.y = dy > 0 ? 1 : -1;
        coda_insere(&se);
    }
}

- (void)mouseDown:(NSEvent *)e
{
    NSPoint pt = [self convertPoint:[e locationInWindow] fromView:nil];
    int mx = (int)pt.x;
    int my = (int)pt.y;
    ph.muris_x = mx;
    ph.muris_y = my;

    pfr_eventus_t se;
    memset(&se, 0, sizeof(se));
    se.typus         = PFR_MURIS_INF;
    se.muris.typus   = PFR_MURIS_INF;
    se.muris.tempus  = pfr_tempus();
    se.muris.x       = mx;
    se.muris.y       = my;
    se.muris.plectrum = 1;
    coda_insere(&se);
}

- (void)rightMouseDown:(NSEvent *)e
{
    NSPoint pt = [self convertPoint:[e locationInWindow] fromView:nil];
    int mx = (int)pt.x;
    int my = (int)pt.y;

    pfr_eventus_t se;
    memset(&se, 0, sizeof(se));
    se.typus         = PFR_MURIS_INF;
    se.muris.typus   = PFR_MURIS_INF;
    se.muris.tempus  = pfr_tempus();
    se.muris.x       = mx;
    se.muris.y       = my;
    se.muris.plectrum = 3;
    coda_insere(&se);
}

- (void)mouseMoved:(NSEvent *)e
{
    NSPoint pt = [self convertPoint:[e locationInWindow] fromView:nil];
    ph.muris_x = (int)pt.x;
    ph.muris_y = (int)pt.y;
}

- (void)mouseDragged:(NSEvent *)e
{
    [self mouseMoved:e];
}

- (void)updateTrackingAreas
{
    [super updateTrackingAreas];
    for (NSTrackingArea *area in self.trackingAreas)
        [self removeTrackingArea:area];
    NSTrackingArea *ta = [[NSTrackingArea alloc]
        initWithRect:self.bounds
             options:(NSTrackingMouseMoved
                    | NSTrackingActiveAlways
                    | NSTrackingInVisibleRect)
               owner:self
            userInfo:nil];
    [self addTrackingArea:ta];
    [ta release];
}

@end

/* ================================================================
 * PhantasmaDelegate — delegatus fenestrae
 * ================================================================ */

@interface PhantasmaDelegate : NSObject <NSWindowDelegate>
@end

@implementation PhantasmaDelegate

- (BOOL)windowShouldClose:(id)sender
{
    (void)sender;
    pfr_eventus_t se;
    memset(&se, 0, sizeof(se));
    se.typus = PFR_EXITUS;
    coda_insere(&se);
    return NO;
}

@end

/* ================================================================
 * pfr_initia / pfr_fini
 * ================================================================ */

int pfr_initia(pfr_u32 flags)
{
    (void)flags;
    memset(&ph, 0, sizeof(ph));

    mach_timebase_info(&ph.tempus_info);
    ph.tempus_initii = mach_absolute_time();

    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    /* menu minimum ut Cmd+Q operetur */
    NSMenu *menu = [[NSMenu alloc] init];
    NSMenuItem *item = [[NSMenuItem alloc] init];
    NSMenu *submenu = [[NSMenu alloc] initWithTitle:@"Phantasma"];
    [submenu addItemWithTitle:@"Quit"
                       action:@selector(terminate:)
                keyEquivalent:@"q"];
    [item setSubmenu:submenu];
    [menu addItem:item];
    [NSApp setMainMenu:menu];
    [submenu release];
    [item release];
    [menu release];

    [NSApp finishLaunching];

    ph.initiatum = 1;
    return 0;
}

void pfr_fini(void)
{
    ph.initiatum = 0;
}

/* ================================================================
 * fenestra
 * ================================================================ */

pfr_fenestra_t *pfr_fenestram_crea(const char *titulus, int x, int y,
                                    int lat, int alt, pfr_u32 flags)
{
    (void)x; (void)y; (void)flags;

    pfr_fenestra_t *f = (pfr_fenestra_t *)calloc(1, sizeof(*f));
    if (!f) {
        snprintf(ph.erratum, sizeof(ph.erratum), "memoria insufficiens");
        return NULL;
    }
    f->latitudo = lat;
    f->altitudo = alt;

    NSRect frame = NSMakeRect(0, 0, lat, alt);
    NSUInteger stilus = NSWindowStyleMaskTitled
                      | NSWindowStyleMaskClosable
                      | NSWindowStyleMaskMiniaturizable;

    f->ns_fenestra = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:stilus
                    backing:NSBackingStoreBuffered
                      defer:NO];
    [f->ns_fenestra setTitle:[NSString stringWithUTF8String:titulus]];
    [f->ns_fenestra center];

    f->ns_visus = [[PhantasmaView alloc] initWithFrame:frame];
    f->ns_visus->lat = lat;
    f->ns_visus->alt = alt;
    [f->ns_fenestra setContentView:f->ns_visus];

    f->ns_delegatus = [[PhantasmaDelegate alloc] init];
    [f->ns_fenestra setDelegate:f->ns_delegatus];

    [f->ns_fenestra makeKeyAndOrderFront:nil];

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [NSApp activateIgnoringOtherApps:YES];
#pragma clang diagnostic pop

    return f;
}

void pfr_fenestram_destrue(pfr_fenestra_t *f)
{
    if (!f) return;
    [f->ns_fenestra setDelegate:nil];
    [f->ns_delegatus release];
    [f->ns_fenestra close];
    free(f);
}

/* ================================================================
 * pictor (renderer)
 * ================================================================ */

pfr_pictor_t *pfr_pictorem_crea(pfr_fenestra_t *f, int index,
                                 pfr_u32 flags)
{
    (void)index; (void)flags;
    if (!f) return NULL;

    pfr_pictor_t *p = (pfr_pictor_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->fenestra = f;
    p->latitudo = f->latitudo;
    p->altitudo = f->altitudo;
    p->alveus = (uint32_t *)calloc((size_t)p->latitudo * p->altitudo,
                                    sizeof(uint32_t));
    if (!p->alveus) { free(p); return NULL; }

    f->ns_visus->alveus = p->alveus;
    return p;
}

void pfr_pictorem_destrue(pfr_pictor_t *p)
{
    if (!p) return;
    if (p->fenestra && p->fenestra->ns_visus)
        p->fenestra->ns_visus->alveus = NULL;
    free(p->alveus);
    free(p);
}

/* ================================================================
 * redditio fenestrae
 * ================================================================ */

void pfr_praesenta(pfr_pictor_t *p)
{
    if (!p || !p->fenestra) return;
    [p->fenestra->ns_visus display];
}

/* ================================================================
 * eventus
 * ================================================================ */

static void mac_eventum_convertere(NSEvent *e)
{
    NSEventType typus = [e type];

    if (typus == NSEventTypeKeyDown || typus == NSEventTypeKeyUp) {
        unsigned short kc = [e keyCode];
        int scancode = (kc < 128) ? mac_scancodex[kc] : 0;
        int symbolum = mac_ad_symbolum(e);
        int depressus = (typus == NSEventTypeKeyDown) ? 1 : 0;

        if (scancode > 0 && scancode < PFR_SC_NUMERUS)
            ph.claves[scancode] = depressus ? 1 : 0;

        pfr_eventus_t se;
        memset(&se, 0, sizeof(se));
        se.typus = depressus ? PFR_CLAVIS_INF : PFR_CLAVIS_SUR;
        se.clavis.typus     = se.typus;
        se.clavis.tempus    = pfr_tempus();
        se.clavis.status    = depressus;
        se.clavis.repetitio = [e isARepeat] ? 1 : 0;
        se.clavis.signum.scancodex = scancode;
        se.clavis.signum.symbolum  = symbolum;
        coda_insere(&se);
    }
    /* rota muris tractatur in PhantasmaView scrollWheel: */
}

int pfr_eventum_lege(pfr_eventus_t *eventus)
{
    @autoreleasepool {
        NSEvent *e;
        while ((e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                       untilDate:nil
                                          inMode:NSDefaultRunLoopMode
                                         dequeue:YES])) {
            mac_eventum_convertere(e);
            [NSApp sendEvent:e];
            [NSApp updateWindows];
        }
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
    uint64_t delta = mach_absolute_time() - ph.tempus_initii;
    uint64_t ns = delta * ph.tempus_info.numer / ph.tempus_info.denom;
    return (pfr_u32)(ns / 1000000ULL);
}
