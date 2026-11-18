#!/usr/bin/env python3
"""Busca la combinación de tomas (una por fragmento) que maximiza la nota total
respetando un jitter global estimado (media ponderada por duración) por debajo del límite.
Después ensambla y evalúa de verdad las mejores combinaciones.

Uso: python3 optimizar_combinacion.py carpeta fragmentos.json [jitter_max=2.95] [top=3]
Requiere los informes informes/fragmento{n}.json generados por elegir_fragmentos.py.
"""
import itertools
import json
import os
import sys

from ensamblar import main as ensamblar
from evaluar import evaluar, resumen

INF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "informes")


def main(carpeta, frag_json, jitter_max="2.95", top="3"):
    jitter_max, top = float(jitter_max), int(top)
    cfg = json.load(open(frag_json, encoding="utf-8"))
    frags, pausas = cfg["fragmentos"], cfg["pausas"]
    cands = []
    for n in range(1, len(frags) + 1):
        res = json.load(open(os.path.join(INF, f"fragmento{n}.json")))
        res = [r for r in res if r["wer"] == 0 and not r["clics"] and not r["pausas_raras"]]
        res.sort(key=lambda r: -r["nota"])
        cands.append(res[:8])
    combos = []
    for c in itertools.product(*cands):
        d = sum(r["duracion_s"] for r in c)
        jit = sum(r["jitter_pct"] * r["duracion_s"] for r in c) / d
        if jit > jitter_max:
            continue
        nis = sum(r["nisqa_tts_media"] * r["duracion_s"] for r in c) / d
        peor = min(r["nisqa_tts_peor_ventana"] for r in c)
        combos.append((2 * nis + 2 * peor - 1 * jit, nis, peor, jit, c))
    combos.sort(key=lambda t: -t[0])
    print(f"{len(combos)} combinaciones cumplen jitter <= {jitter_max}")
    mejores = []
    for i, (s, nis, peor, jit, c) in enumerate(combos[:top]):
        nombre = f"combo{os.environ.get("COMBO_TAG", "")}{i + 1}"
        plan = {"fragmentos": [{"wav": os.path.join(carpeta, r["archivo"]), "pausa_despues": p}
                               for r, p in zip(c, pausas + [0])]}
        plan_path = os.path.join(carpeta, "..", f"{nombre}_plan.json")
        json.dump(plan, open(plan_path, "w"), ensure_ascii=False, indent=1)
        out = os.path.join(carpeta, "..", f"{nombre}.wav")
        ensamblar(plan_path, out)
        r = evaluar(out, " ".join(frags), png=os.path.join(INF, f"{nombre}.png"))
        json.dump(r, open(os.path.join(INF, f"{nombre}.json"), "w"), ensure_ascii=False, indent=1, default=float)
        print(f"{nombre}: {[x['archivo'] for x in c]} estimado nisqa={nis:.2f} jit={jit:.2f}")
        print("   REAL:", resumen(r))
        print("   Fallos:", r["fallos"] or "NINGUNO -> APROBADO", flush=True)
        mejores.append((r, out))
    return mejores


if __name__ == "__main__":
    main(*sys.argv[1:])
