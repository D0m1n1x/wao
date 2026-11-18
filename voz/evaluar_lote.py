#!/usr/bin/env python3
"""Evalúa todas las tomas .wav de una carpeta y las ordena por una puntuación compuesta.

Uso: python3 evaluar_lote.py carpeta guion.txt [informes_dir]
"""
import glob
import json
import os
import sys

from evaluar import evaluar, resumen


def puntuacion(r):
    """Más alto = mejor. Naturalidad manda; errores de texto y fallos penalizan fuerte."""
    s = 2.0 * r["nisqa_tts_media"] + 1.0 * r["nisqa_tts_peor_ventana"]
    s += 1.0 * r["dnsmos_ovrl"] + 0.5 * r["dnsmos_sig"] + 0.5 * r["dnsmos_p808"]
    s -= 10.0 * r["wer"]
    s -= 0.25 * len(r["fallos"])
    return round(s, 3)


def main(carpeta, guion_path, inf_dir=None):
    guion = open(guion_path, encoding="utf-8").read().strip()
    inf_dir = inf_dir or os.path.join(os.path.dirname(carpeta.rstrip("/")), "..", "informes")
    os.makedirs(inf_dir, exist_ok=True)
    res = []
    for wav in sorted(glob.glob(os.path.join(carpeta, "*.wav"))):
        base = os.path.splitext(os.path.basename(wav))[0]
        r = evaluar(wav, guion, png=os.path.join(inf_dir, base + ".png"))
        r["puntuacion"] = puntuacion(r)
        json.dump(r, open(os.path.join(inf_dir, base + ".json"), "w"), ensure_ascii=False, indent=1, default=float)
        res.append(r)
        print(f"{base}: score={r['puntuacion']} | {resumen(r)}", flush=True)
        print(f"   fallos: {r['fallos'] or 'ninguno'}", flush=True)
        if r.get("errores"):
            print(f"   texto: {r['errores']}", flush=True)
    res.sort(key=lambda r: -r["puntuacion"])
    print("\nRANKING")
    for r in res:
        print(f"{r['puntuacion']:7.3f}  {r['archivo']:24s} fallos={len(r['fallos'])} aprobado={r['aprobado']}")
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
