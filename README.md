# Phantasma

A complete, self-contained graphics library that gives you a window, a renderer, keyboard and mouse input, animated GIF encoding, and H.264 MP4 video encoding — all in a single static library with absolutely zero external dependencies.

Phantasma replaces SDL2, libx264, giflib, and ffmpeg in one shot. No package managers. No dynamic linking. No configuration scripts. No hunting down development headers. One library, one header, one include path, and you're rendering interactive graphics and recording video.

## What You Get

A minimal, razor-sharp API designed for exactly the kind of program that needs a window, a pixel buffer, and a way to record what's on screen. The entire public interface fits in a single header file and consists of exactly the functions you need — not a single one more.

**Windowing and rendering.** Create a window, push ARGB8888 pixel buffers to it, draw points and filled rectangles on top, present the frame. Hardware-accelerated where possible, software-rendered everywhere. Works identically on macOS (Cocoa + Core Graphics) and Linux (Xlib).

**Input handling.** Poll-based event loop with discrete key events and continuous keyboard state queries. Mouse wheel support. Window close detection. The event model is clean, predictable, and designed for real-time interactive applications.

**GIF recording.** Feed ARGB frames in, get a beautifully dithered animated GIF out. Median-cut color quantization to 128 colors, Bayer 8x8 ordered dithering, LZW compression, automatic downscaling, infinite looping — all built in. The output is compact and looks excellent.

**MP4 recording.** Feed ARGB frames in, get a valid H.264 Baseline MP4 out. Proper NAL unit packaging, correct SPS/PPS generation, full MP4 container with moov/mdat structure. Every frame is an IDR — seek to any point instantly.

## Building

```bash
face
```

That produces `libphantasma.a`. Link it:

```bash
cc program.o libphantasma.a -framework Cocoa -o program    # macOS
cc program.o libphantasma.a -lX11 -lm -o program           # Linux
```

## Quick Start

```c
#include "phantasma.h"

pfr_initia(PFR_INITIA_VIDEO);

pfr_fenestra_t *f = pfr_fenestram_crea("Title",
    PFR_POS_MEDIUM, PFR_POS_MEDIUM, 640, 480, 0);
pfr_pictor_t *p = pfr_pictorem_crea(f, -1, PFR_PICTOR_CELER);
pfr_textura_t *t = pfr_texturam_crea(p, PFR_PIXEL_ARGB8888,
    PFR_TEXTURA_FLUENS, 640, 480);

while (running) {
    pfr_eventus_t ev;
    while (pfr_eventum_lege(&ev))
        if (ev.typus == PFR_EXITUS) running = 0;

    pfr_texturam_renova(t, NULL, pixels, 640 * 4);
    pfr_purga(p);
    pfr_texturam_pinge(p, t, NULL, NULL);
    pfr_praesenta(p);
}
```

Recording a GIF is three function calls:

```c
pfr_gif_t *g = pfr_gif_initia("out.gif", 768, 768, 3, 2);
/* for each frame: */
pfr_gif_tabulam_adde(g, pixels);
/* when done: */
pfr_gif_fini(g);
```

Recording an MP4 is three function calls:

```c
pfr_mp4_t *m = pfr_mp4_initia("out.mp4", 768, 768, 30);
/* for each frame: */
pfr_mp4_tabulam_adde(m, pixels);
/* when done: */
pfr_mp4_fini(m);
```

## Zero Dependencies

The entire library compiles with nothing but a C compiler and system headers. On macOS it uses Cocoa, which ships with every Mac since 2001. On Linux it uses Xlib, which ships with every X11 installation since 1987. The GIF encoder implements LZW compression from scratch. The MP4 encoder implements H.264 NAL unit generation, bitstream formatting, and ISO base media file container writing from scratch. No generated code, no embedded blobs, no submodules, no vendored libraries. Every byte of this was written by hand.

## License

Free. Public domain. Use however you like.
