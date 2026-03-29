# Phantasma

Bibliotheca fenestrae et inscriptionis minimalis sine ullis dependentiis externis.

> **Nota:** README.md ignorandum est — plicam illam scriptores Anglici legunt. Haec plica sola documentatio vera est.

## Aedificatio

```
face
```

Linux:

```
face -f Faceplica.linux
```

Bibliotheca statica `libphantasma.a` generatur.

## Nexus

Darwin:

```
cc programma.o libphantasma.a -framework Cocoa -o programma
```

Linux:

```
cc programma.o libphantasma.a -lX11 -lm -o programma
```

## Usus

### Fenestra et redditio

```
pfr_initia(PFR_INITIA_VIDEO);

pfr_fenestra_t *f = pfr_fenestram_crea("Titulus",
    PFR_POS_MEDIUM, PFR_POS_MEDIUM, 640, 480, 0);
pfr_pictor_t *p = pfr_pictorem_crea(f, -1, PFR_PICTOR_CELER);
pfr_textura_t *t = pfr_texturam_crea(p, PFR_PIXEL_ARGB8888,
    PFR_TEXTURA_FLUENS, 640, 480);

pfr_texturam_renova(t, NULL, pixels, 640 * 4);
pfr_purga(p);
pfr_texturam_pinge(p, t, NULL, NULL);
pfr_praesenta(p);
```

### Eventus

```
pfr_eventus_t ev;
while (pfr_eventum_lege(&ev)) {
    if (ev.typus == PFR_EXITUS) currit = 0;
    if (ev.typus == PFR_CLAVIS_INF)
        int sym = ev.clavis.signum.symbolum;
    if (ev.typus == PFR_ROTA_MURIS)
        int dy = ev.rota.y;
}

const pfr_u8 *claves = pfr_claves_status(NULL);
if (claves[PFR_SC_W]) /* clavis W depressa */;
```

### GIF inscriptor

```
pfr_gif_t *g = pfr_gif_initia("via.gif", lat, alt, 3, 2);
pfr_gif_tabulam_adde(g, pixels_argb);
pfr_gif_fini(g);
```

Parametra: via, latitudo, altitudo, mora in centisecundis, factor scalae (2 = dimidia).

Median-cut quantisatio ad 128 colores. Bayer dithering. LZW compressio.

### MP4 inscriptor

```
pfr_mp4_t *m = pfr_mp4_initia("via.mp4", lat, alt, 30);
pfr_mp4_tabulam_adde(m, pixels_argb);
pfr_mp4_fini(m);
```

H.264 Baseline. Omnes tabulae IDR. I-PCM modus.

## Plicae

| Plica | Descriptio |
|---|---|
| `phantasma.h` | Caput publicum |
| `phantasma_darwin.m` | Implementatio macOS (Cocoa) |
| `phantasma_linux.c` | Implementatio Linux (X11) |
| `pfr_gif.c` | GIF inscriptor |
| `pfr_mp4.c` | MP4 inscriptor |

## Dependentiae

Nullae dependentiae externae. Darwin: Cocoa (systema). Linux: Xlib (systema).

## Licentia

Idem ac apotheca.
