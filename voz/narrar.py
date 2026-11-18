#!/usr/bin/env python3
"""Narra un guion completo con la voz fijada del canal, con control de calidad automático.

    python3 narrar.py guion.txt salida/                 # 6 tomas por fragmento (máxima calidad)
    python3 narrar.py guion.txt salida/ --tomas 3       # más rápido
    python3 narrar.py guion.txt salida2/ --semilla 1    # otro intento con tomas distintas

Proceso (el mismo que se hizo a mano para la locución de prueba):
  1. Prepara el texto (números y símbolos a palabras) y lo divide en fragmentos de 1-3 frases.
  2. Genera N tomas de cada fragmento con la voz fijada (voz_fijada/narrador_F_ref.wav).
  3. Evalúa cada toma: palabras exactas (Whisper), naturalidad (NISQA), distorsión (DNSMOS),
     aspereza (jitter/HNR), entonación (picos, vocal fry), pausas raras y clics.
  4. Si ninguna toma de un fragmento pasa el control, genera más ("si no sale, vuelve a intentarlo").
  5. Elige la combinación más natural que mantenga el jitter global bajo el umbral.
  6. Ensambla con pausas naturales, masteriza de forma transparente y evalúa el resultado completo.
     Si el resultado falla, rehace los fragmentos culpables y repite.

Salida: salida/narracion.wav (48 kHz/24 bit), salida/narracion.mp3, salida/informe.txt, salida/informe.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))


def cargar_entorno():
    """Lee las rutas de modelos que dejó instalar.sh."""
    f = os.path.join(AQUI, ".entorno.json")
    if os.path.exists(f):
        for k, v in json.load(open(f)).items():
            os.environ.setdefault(k, v)
    faltan = [k for k in ("QWEN_BASE", "WHISPER_MODEL", "VOZ_MODELS") if not os.environ.get(k)]
    if faltan:
        sys.exit(f"Faltan modelos {faltan}: ejecuta primero  bash {os.path.join(AQUI, 'instalar.sh')}")


cargar_entorno()
sys.path.insert(0, AQUI)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from ensamblar import main as ensamblar  # noqa: E402
from evaluar import evaluar, normaliza, resumen  # noqa: E402

REF_WAV = os.path.join(AQUI, "voz_fijada", "narrador_F_ref.wav")
REF_TXT = open(os.path.join(AQUI, "voz_fijada", "narrador_F_ref.txt"), encoding="utf-8").read().strip()

# Control de calidad de cada fragmento (calibrado con voz humana real, ver README)
PUERTA = dict(nisqa_min=3.9, jitter_max=3.4, hnr_min=10.5, picos_max=4.0, fry_max=3.0)
JITTER_GLOBAL = 2.97


# ----------------------------------------------------------------------------- texto
def preparar_texto(t):
    from num2words import num2words

    t = t.replace("\r", "")
    t = re.sub(r"(\d+)\s?%", lambda m: m.group(1) + " por ciento", t)
    t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: f"{m.group(1)} y {int(m.group(2))}" if m.group(2) != "00" else m.group(1), t)
    t = re.sub(r"\b(\d{1,3})((?:\.\d{3})+)\b", lambda m: m.group(1) + m.group(2).replace(".", ""), t)  # 2.500 -> 2500
    t = re.sub(r"\$\s?(\d+)", r"\1 dólares", t)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="es"), t)
    t = t.replace("…", "...").replace("&", " y ")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def fragmentar(texto, max_chars=170, min_chars=45):
    """Divide en fragmentos de frases completas. Devuelve [(texto, pausa_despues_s)]."""
    frags = []
    parrafos = [" ".join(p.split()) for p in re.split(r"\n\s*\n", texto) if p.strip()]
    for pi, p in enumerate(parrafos):
        # fin de frase = . ? ! (o ...) seguido de mayúscula, ¿, ¡ o comillas; "teléfono... pero" NO corta
        oraciones = [o.strip() for o in re.split(r"(?<=[.?!])[\"»”)]?\s+(?=[¿¡\"«“A-ZÁÉÍÓÚÑ])", p) if o.strip()]
        grupo = []
        for o in oraciones:
            if grupo and len(" ".join(grupo + [o])) > max_chars:
                frags.append([" ".join(grupo), False])
                grupo = []
            grupo.append(o)
        if grupo:
            if frags and len(" ".join(grupo)) < min_chars and not frags[-1][1] and \
                    len(frags[-1][0]) + len(" ".join(grupo)) < max_chars * 1.4:
                frags[-1][0] += " " + " ".join(grupo)  # no dejar un fragmento final diminuto
            else:
                frags.append([" ".join(grupo), False])
        frags[-1][1] = True  # fin de párrafo
    out = []
    for t, fin_parrafo in frags:
        if fin_parrafo:
            pausa = 1.0
        elif t.endswith("..."):
            pausa = 0.75
        elif t.endswith(("?", "!")):
            pausa = 0.65
        else:
            pausa = 0.6
        out.append((t, pausa))
    if out:
        out[-1] = (out[-1][0], 0.0)
    return out


# ----------------------------------------------------------------------------- generación
class Narrador:
    def __init__(self, ruta_modelo):
        from qwen_tts import Qwen3TTSModel

        gpu = torch.cuda.is_available()
        torch.set_num_threads(os.cpu_count() or 4)
        t0 = time.time()
        self.m = Qwen3TTSModel.from_pretrained(ruta_modelo, device_map="cuda:0" if gpu else "cpu",
                                               dtype=torch.bfloat16, attn_implementation="sdpa")
        self.prompt = self.m.create_voice_clone_prompt(ref_audio=REF_WAV, ref_text=REF_TXT)
        log(f"Modelo de voz cargado en {time.time() - t0:.0f}s ({'GPU' if gpu else 'CPU'})")

    def tomas(self, texto, n, semilla):
        torch.manual_seed(semilla)
        np.random.seed(semilla % (2 ** 32))
        prompt = self.prompt * n if isinstance(self.prompt, list) else self.prompt
        wavs, sr = self.m.generate_voice_clone(text=[texto] * n, language=["Spanish"] * n,
                                               voice_clone_prompt=prompt, subtalker_temperature=0.6)
        return wavs, sr


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nota(r):
    s = 2.0 * r["nisqa_tts_media"] + 1.0 * r["nisqa_tts_peor_ventana"]
    s += 1.0 * r["dnsmos_ovrl"] + 0.5 * r["dnsmos_sig"] + 0.5 * r["dnsmos_p808"]
    s -= 10.0 * r["wer"] + 2.0 * max(0.0, r["jitter_pct"] - 2.8) + 0.5 * max(0.0, 10.5 - r["hnr_db"])
    s -= 0.3 * max(0.0, r["f0_picos_pct"] - 4.0) + 0.5 * max(0.0, r["f0_fry_pct"] - 3.0)
    s -= 1.0 * r["pausas_raras"] + 1.0 * r["clics"] + 1.0 * (r["clipping_muestras"] > 0)
    return round(s, 3)


def pasa(r):
    return (r["wer"] == 0 and r["clics"] == 0 and r["pausas_raras"] == 0 and r["clipping_muestras"] == 0
            and r["nisqa_tts_media"] >= PUERTA["nisqa_min"] and r["jitter_pct"] <= PUERTA["jitter_max"]
            and r["hnr_db"] >= PUERTA["hnr_min"] and r["f0_picos_pct"] <= PUERTA["picos_max"]
            and r["f0_fry_pct"] <= PUERTA["fry_max"])


SEMILLA_BASE = 0


def generar_fragmento(narr, i, texto, n, carpeta, cands, ronda):
    t0 = time.time()
    wavs, sr = narr.tomas(texto, n, semilla=SEMILLA_BASE * 1_000_003 + 1000 * (i + 1) + 97 * ronda)
    for k, w in enumerate(wavs):
        ruta = os.path.join(carpeta, f"f{i + 1:03d}_r{ronda}_t{k + 1}.wav")
        sf.write(ruta, w, sr, subtype="PCM_24")
        r = evaluar(ruta, texto)
        r.update(ruta=ruta, nota=nota(r), pasa=pasa(r))
        r.pop("pausas", None)
        cands[i].append(r)
    ok = sum(r["pasa"] for r in cands[i])
    mejor = max(cands[i], key=lambda r: r["nota"])
    log(f"  fragmento {i + 1}: ronda {ronda}, {n} tomas en {time.time() - t0:.0f}s -> {ok} aprobadas "
        f"(mejor nisqa={mejor['nisqa_tts_media']:.2f} jitter={mejor['jitter_pct']:.2f} wer={mejor['wer']:.2f})")


def elegir(cands, jitter_max):
    """Mejor toma por fragmento y, si el jitter global estimado se pasa, cambia las tomas
    que más jitter quitan perdiendo menos nota."""
    pool = [[r for r in c if r["pasa"]] or sorted(c, key=lambda r: -r["nota"])[:1] for c in cands]
    sel = [max(p, key=lambda r: r["nota"]) for p in pool]

    def jit(s):
        return sum(r["jitter_pct"] * r["duracion_s"] for r in s) / sum(r["duracion_s"] for r in s)

    while jit(sel) > jitter_max:
        mejor = None
        for i, p in enumerate(pool):
            for r in p:
                dj = (sel[i]["jitter_pct"] - r["jitter_pct"]) * r["duracion_s"]
                if dj <= 0:
                    continue
                coste = (sel[i]["nota"] - r["nota"]) / dj
                if mejor is None or coste < mejor[0]:
                    mejor = (coste, i, r)
        if mejor is None:
            break
        sel[mejor[1]] = mejor[2]
    return sel, jit(sel)


def culpables(final, tramos, frags):
    """Fragmentos responsables de un fallo del montaje completo."""
    malos = set()
    if final["nisqa_tts_peor_ventana"] < 3.6:
        for t, v in final.get("nisqa_ventanas", []):
            if v < 3.6:
                malos |= {i for i, (a, b) in enumerate(tramos) if a < t + 6 and b > t}
    if final["wer"] > 0:
        limites, acc = [], 0
        for t, _ in frags:
            acc += len(normaliza(t))
            limites.append(acc)
        for a0, _ in final.get("errores_idx", []):
            malos.add(next((i for i, l in enumerate(limites) if a0 < l), len(frags) - 1))
    return sorted(malos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("guion")
    ap.add_argument("salida")
    ap.add_argument("--tomas", type=int, default=6, help="tomas por fragmento y ronda (6 = máxima calidad)")
    ap.add_argument("--rondas", type=int, default=4, help="rondas máximas por fragmento si ninguna toma aprueba")
    ap.add_argument("--revisiones", type=int, default=2, help="veces que se rehace el montaje si falla")
    ap.add_argument("--semilla", type=int, default=0, help="cámbiala para obtener tomas distintas al repetir")
    a = ap.parse_args()
    global SEMILLA_BASE
    SEMILLA_BASE = a.semilla

    os.makedirs(a.salida, exist_ok=True)
    carpeta = os.path.join(a.salida, "tomas")
    os.makedirs(carpeta, exist_ok=True)
    texto = preparar_texto(open(a.guion, encoding="utf-8").read())
    frags = fragmentar(texto)
    json.dump({"fragmentos": [t for t, _ in frags], "pausas": [p for _, p in frags]},
              open(os.path.join(a.salida, "fragmentos.json"), "w"), ensure_ascii=False, indent=1)
    log(f"Guion: {len(texto)} caracteres -> {len(frags)} fragmentos")
    for i, (t, p) in enumerate(frags, 1):
        log(f"  {i:3d}. ({p:.2f}s) {t}")

    narr = Narrador(os.environ["QWEN_BASE"])
    cands = [[] for _ in frags]
    rondas = [0] * len(frags)

    def trabajar(i, max_rondas):
        while rondas[i] < max_rondas:
            rondas[i] += 1
            generar_fragmento(narr, i, frags[i][0], a.tomas, carpeta, cands, rondas[i])
            if any(r["pasa"] for r in cands[i]):
                return
        log(f"  AVISO fragmento {i + 1}: ninguna toma aprobó tras {rondas[i]} rondas; se usa la mejor")

    for i in range(len(frags)):
        trabajar(i, a.rondas)

    final, tramos = None, None
    for rev in range(a.revisiones + 1):
        sel, jit = elegir(cands, JITTER_GLOBAL)
        plan = {"fragmentos": [{"wav": r["ruta"], "pausa_despues": p} for r, (_, p) in zip(sel, frags)]}
        plan_path = os.path.join(a.salida, "plan.json")
        json.dump(plan, open(plan_path, "w"), ensure_ascii=False, indent=1)
        bruto = os.path.join(a.salida, "ensamblado_24k.wav")
        tramos = ensamblar(plan_path, bruto)
        base = os.path.join(a.salida, "narracion")
        subprocess.run(["bash", os.path.join(AQUI, "masterizar.sh"), bruto, base], check=True)
        final = evaluar(base + ".wav", " ".join(t for t, _ in frags), png=os.path.join(a.salida, "espectrograma.png"))
        log(f"Montaje {rev + 1}: {resumen(final)}")
        log(f"  Fallos: {final['fallos'] or 'NINGUNO -> APROBADO'}")
        if not final["fallos"]:
            break
        malos = culpables(final, [(a0 + 0.3, b0 + 0.3) for a0, b0 in tramos], frags)  # +0,3 s del master
        if not malos or rev == a.revisiones:
            break
        log(f"  Rehaciendo fragmentos {[m + 1 for m in malos]}")
        for i in malos:
            trabajar(i, rondas[i] + 2)

    rep = {k: v for k, v in final.items() if k not in ("nisqa_ventanas",)}
    rep["fragmentos"] = [{"texto": t, "toma": os.path.basename(r["ruta"]), "nisqa": r["nisqa_tts_media"],
                          "jitter": r["jitter_pct"]} for (t, _), r in zip(frags, sel)]
    json.dump(rep, open(os.path.join(a.salida, "informe.json"), "w"), ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(a.salida, "informe.txt"), "w", encoding="utf-8") as f:
        f.write("APROBADO\n" if not final["fallos"] else f"CON AVISOS: {final['fallos']}\n")
        f.write(resumen(final) + "\n\nTranscripción de control:\n" + final.get("transcripcion", "") + "\n")
    log(f"Listo: {base}.mp3  ({final['duracion_s']:.1f}s)  {'APROBADO' if not final['fallos'] else 'CON AVISOS'}")


if __name__ == "__main__":
    main()
