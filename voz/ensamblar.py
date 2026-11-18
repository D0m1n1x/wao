#!/usr/bin/env python3
"""Ensambla una locución a partir de las mejores tomas de cada fragmento ("punch-in").

- Recorta el silencio sobrante de cada fragmento (deja 40 ms de aire).
- Iguala la sonoridad de cada fragmento a la mediana del conjunto (solo si difiere > 0,5 LU).
- Inserta pausas con la duración indicada (un narrador real hace pausas de 0,4-0,9 s entre frases).
- Fundidos de 10 ms en cada borde para que no haya clics.

Uso: python3 ensamblar.py plan.json salida.wav
plan.json = {"fragmentos": [{"wav": "...", "pausa_despues": 0.6}, ...]}
"""
import json
import sys

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def recorta(y, sr, umbral_db=-45, aire=0.04):
    db = librosa.amplitude_to_db(np.abs(y) + 1e-9, ref=np.max(np.abs(y)))
    idx = np.where(db > umbral_db)[0]
    a = max(0, idx[0] - int(aire * sr))
    b = min(len(y), idx[-1] + int(aire * sr))
    return y[a:b]


def fundido(y, sr, ms=10):
    n = int(sr * ms / 1000)
    r = np.linspace(0, 1, n)
    y = y.copy()
    y[:n] *= r
    y[-n:] *= r[::-1]
    return y


def main(plan_path, salida):
    plan = json.load(open(plan_path, encoding="utf-8"))
    frs, sr0 = [], None
    for f in plan["fragmentos"]:
        y, sr = sf.read(f["wav"])
        if y.ndim > 1:
            y = y.mean(axis=1)
        sr0 = sr0 or sr
        assert sr == sr0, "todas las tomas deben tener la misma frecuencia de muestreo"
        frs.append((fundido(recorta(y, sr), sr), float(f.get("pausa_despues", 0.6))))
    meter = pyln.Meter(sr0)
    louds = [meter.integrated_loudness(y) for y, _ in frs]
    objetivo = float(np.median(louds))
    out, tramos, pos = [], [], 0
    for (y, pausa), L in zip(frs, louds):
        if abs(L - objetivo) > 0.5:
            y = y * 10 ** ((objetivo - L) / 20)
        out.append(y)
        tramos.append((pos / sr0, (pos + len(y)) / sr0))
        pos += len(y) + int(pausa * sr0)
        out.append(np.zeros(int(pausa * sr0)))
    y = np.concatenate(out[:-1])
    pico = np.max(np.abs(y))
    if pico > 0.89:
        y *= 0.89 / pico
    sf.write(salida, y, sr0, subtype="PCM_24")
    print(f"OK {salida} {len(y) / sr0:.1f}s | sonoridad por fragmento: {[round(float(x), 1) for x in louds]}")
    return tramos


if __name__ == "__main__":
    main(*sys.argv[1:])
