#!/bin/sh
# nexus_genera.sh — genera flagella nexus pro platforma
#
# Variabiles: CC, CFLAGS (a Faceplica per env traduntur)

if echo "$CFLAGS" | grep -q PHANTASMA_X11; then
    echo "-lX11"
elif $CC -E -dM - </dev/null 2>/dev/null | grep -q __APPLE__; then
    if which nvcc >/dev/null 2>&1; then
        echo "-framework Cocoa -framework Metal -framework Foundation -lcudart"
    else
        echo "-framework Cocoa -framework Metal -framework Foundation"
    fi
else
    if which nvcc >/dev/null 2>&1; then
        echo "-lX11 -lcudart"
    else
        echo "-lX11"
    fi
fi
