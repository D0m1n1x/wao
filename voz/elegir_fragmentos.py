#!/usr/bin/env python3
"""Punch-in: elige la mejor toma de cada fragmento y ensambla la locución completa.

Uso: python3 elegir_fragmentos.py carpeta_tomas fragmentos.json salida.wav
Las tomas deben llamarse f{n}_t{k}.wav (fragmento n, toma k).
"""
import glob
import json
import os
import re
import sys

from evaluar import evaluar, resumen

INF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "informes")


def nota_fragmento(r):
    s = 2.0 * r["nisqa_tts_media"] + 1.0 * r["nisqa_tts_peor_ventana"]
    s += 1.0 * r["dnsmos_ovrl"] + 0.5 * r["dnsmos_sig"] + 0.5 * r["dnsmos_p808"]
    s -= 10.0 * r["wer"]                                   # cualquier palabra mal = descartada en la práctica
    s -= 2.0 * max(0.0, r["jitter_pct"] - 2.8)             # aspereza
    s -= 0.5 * max(0.0, 10.5 - r["hnr_db"])                # voz "sucia"
    s -= 0.3 * max(0.0, r["f0_picos_pct"] - 4.0)           # entonación exagerada
    s -= 0.5 * max(0.0, r["f0_fry_pct"] - 3.0)             # carraspeo
    s -= 1.0 * r["pausas_raras"] + 1.0 * r["clics"] + 1.0 * (r["clipping_muestras"] > 0)
    return round(s, 3)


def main(carpeta, frag_json, salida):
    cfg = json.load(open(frag_json, encoding="utf-8"))
    frags, pausas = cfg["fragmentos"], cfg["pausas"]
    elegidas = []
    for n, texto in enumerate(frags, 1):
        tomas = sorted(glob.glob(os.path.join(carpeta, f"f{n}_t*.wav")),
                       key=lambda p: int(re.search(r"_t(\d+)", p).group(1)))
        res = []
        for wav in tomas:
            r = evaluar(wav, texto)
            r["nota"] = nota_fragmento(r)
            res.append(r)
            print(f"  f{n} {os.path.basename(wav)} nota={r['nota']} wer={r['wer']:.2f} nisqa={r['nisqa_tts_media']:.2f}"
                  f" jit={r['jitter_pct']:.2f} hnr={r['hnr_db']:.1f} picos={r['f0_picos_pct']:.1f}"
                  f" {('ERR ' + str(r['errores'])) if r['errores'] else ''}", flush=True)
        mejor = max(res, key=lambda r: r["nota"])
        print(f"f{n} -> {mejor['archivo']} (nota {mejor['nota']})", flush=True)
        elegidas.append(os.path.join(carpeta, mejor["archivo"]))
        json.dump(res, open(os.path.join(INF, f"fragmento{n}.json"), "w"), ensure_ascii=False, indent=1, default=float)
    plan = {"fragmentos": [{"wav": w, "pausa_despues": p} for w, p in zip(elegidas, pausas + [0])]}
    plan_path = os.path.splitext(salida)[0] + "_plan.json"
    json.dump(plan, open(plan_path, "w"), ensure_ascii=False, indent=1)
    from ensamblar import main as ensamblar
    ensamblar(plan_path, salida)
    guion = " ".join(frags)
    r = evaluar(salida, guion, png=os.path.join(INF, os.path.splitext(os.path.basename(salida))[0] + ".png"))
    json.dump(r, open(os.path.join(INF, os.path.splitext(os.path.basename(salida))[0] + ".json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    print("\nENSAMBLADO:", resumen(r))
    for p in r.get("pausas", []):
        print(f"  pausa {p['dur']:.2f}s tras '{p['tras']}{p['punt']}'{'  <-- RARA' if p['rara'] else ''}")
    print("Errores de texto:", r.get("errores") or "ninguno")
    print("Fallos:", r["fallos"] or "NINGUNO -> APROBADO")


if __name__ == "__main__":
    main(*sys.argv[1:])
