/*
 * phantasma_plat.c — involucrum platformae
 *
 * Defini PHANTASMA_X11 si vis X11 uti in Darwin.
 */

#if defined(__APPLE__) && !defined(PHANTASMA_X11)
#include "phantasma_darwin.m"
#elif defined(__linux__) || defined(PHANTASMA_X11)
#include "phantasma_linux.c"
#else
#error "tergum ignotum — defini PHANTASMA_X11 si vis X11 uti"
#endif
