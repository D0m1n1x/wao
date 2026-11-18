#!/usr/bin/env python3
"""Genera tomas de voz con Qwen3-TTS (Apache-2.0) en local.

Modos:
  design : crea la voz a partir de una descripción en lenguaje natural (VoiceDesign 1.7B).
  clone  : reutiliza una voz ya diseñada (clip de referencia + su texto) con el modelo Base 1.7B,
           para que todas las tomas suenen al MISMO narrador.

Uso:
  python3 generar.py trabajos.json
trabajos.json = {"modo": "design"|"clone", "texto": "...", "salida_dir": "...",
                 "ref_audio": "...", "ref_text": "...",
                 "tomas": [{"nombre": "...", "instruct": "...", "seed": 1, "temperature": 0.9, "top_p": 1.0}, ...]}
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

torch.set_num_threads(os.cpu_count() or 4)

MODELOS = {
    "design": os.environ.get("QWEN_DESIGN", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "clone": os.environ.get("QWEN_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
}


def main(cfg_path):
    from qwen_tts import Qwen3TTSModel

    cfg = json.load(open(cfg_path, encoding="utf-8"))
    modo = cfg["modo"]
    out_dir = cfg["salida_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(MODELOS[modo], device_map="cpu", dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
    print(f"modelo {modo} cargado en {time.time() - t0:.0f}s", flush=True)

    prompt = None
    if modo == "clone":
        prompt = model.create_voice_clone_prompt(ref_audio=cfg["ref_audio"], ref_text=cfg["ref_text"])

    lote = int(cfg.get("lote", 1))
    tomas = cfg["tomas"]
    for i in range(0, len(tomas), lote):
        grupo = tomas[i:i + lote]
        seed = int(grupo[0].get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)
        textos = [t.get("texto", cfg.get("texto")) for t in grupo]
        kw = {k: grupo[0][k] for k in ("temperature", "top_p", "top_k", "repetition_penalty",
                                        "subtalker_temperature", "subtalker_top_p", "subtalker_top_k") if k in grupo[0]}
        t1 = time.time()
        if modo == "design":
            wavs, sr = model.generate_voice_design(text=textos, instruct=[t["instruct"] for t in grupo],
                                                   language=["Spanish"] * len(grupo), **kw)
        else:
            wavs, sr = model.generate_voice_clone(text=textos, language=["Spanish"] * len(grupo),
                                                  voice_clone_prompt=prompt * len(grupo) if isinstance(prompt, list) else prompt,
                                                  **kw)
        total = 0.0
        for t, w in zip(grupo, wavs):
            path = os.path.join(out_dir, t["nombre"] + ".wav")
            sf.write(path, w, sr, subtype="PCM_24")
            total += len(w) / sr
            print(f"OK {path} {len(w) / sr:.1f}s", flush=True)
        el = time.time() - t1
        print(f"lote de {len(grupo)}: {total:.1f}s de audio en {el:.0f}s (RTF {el / max(total, 1e-3):.2f})", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
