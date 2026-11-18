#!/usr/bin/env python3
"""Evaluador objetivo de una locución TTS.

Mide lo que un oído exigente notaría:
  - Naturalidad (NISQA-TTS, MOS 1-5) global y por ventanas de 6 s (la peor ventana delata fallos locales).
  - Distorsión / ruido (DNSMOS P.835: SIG, BAK, OVRL y P.808).
  - Exactitud del texto (Whisper large-v3-turbo -> WER contra el guion).
  - Entonación: variación de F0 en semitonos (monótono = robótico), jitter, shimmer, HNR.
  - Ritmo: sílabas/segundo y pausas (¿caen en la puntuación? ¿hay pausas raras a mitad de frase?).
  - Técnica: picos, clipping, clics, sonoridad (LUFS), silencios al inicio/fin.

Uso: python3 evaluar.py audio.wav guion.txt [--png espectro.png] [--json salida.json]
Variables de entorno: VOZ_MODELS (carpeta con DNSMOS/NISQA), WHISPER_MODEL (ruta o alias).
"""
import argparse
import json
import math
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher

import librosa
import numpy as np
import soundfile as sf

MODELS = os.environ.get("VOZ_MODELS", "")
WHISPER = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

# Umbrales de "perfecto". Una toma solo se acepta si los cumple todos.
# Calibrados con voz humana real (LibriSpeech, narrador masculino): jitter 2,2-2,6 %, HNR 10-11 dB,
# F0 std 3,1-3,7 st, picos de F0 hasta 4 %, NISQA 3,4-3,9, DNSMOS SIG 3,7.
UMBRALES = {
    "nisqa_tts_media": (">=", 4.0),
    "nisqa_tts_peor_ventana": (">=", 3.6),
    "dnsmos_sig": (">=", 3.4),
    "dnsmos_bak": (">=", 4.0),
    "dnsmos_ovrl": (">=", 3.2),
    "wer": ("<=", 0.0),
    "f0_std_st": (">=", 2.5),
    "f0_rango_st": (">=", 8.0),
    "jitter_pct": ("<=", 3.0),
    "f0_picos_pct": ("<=", 4.0),
    "f0_fry_pct": ("<=", 3.0),
    "salto_volumen_db": ("<=", 9.0),
    "hnr_db": (">=", 10.0),
    "silabas_por_seg": ("rango", (4.5, 7.5)),  # español leído ~7,8 síl/s (Pellegrino et al. 2011)
    "pausas_raras": ("<=", 0),
    "pausa_max_s": ("<=", 1.3),
    "pausa_media_s": ("rango", (0.35, 0.95)),
    "pausas_puntuacion_faltantes": ("<=", 1),
    "clipping_muestras": ("<=", 0),
    "clics": ("<=", 0),
    "silencio_inicio_s": ("<=", 0.8),
    "silencio_final_s": ("<=", 1.5),
}


# ----------------------------------------------------------------------------- texto
def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _num_es(m):
    from num2words import num2words
    return " " + num2words(int(m.group(0)), lang="es") + " "


def normaliza(s):
    s = re.sub(r"(\d+):(\d+)", r"\1 y \2", s)
    s = re.sub(r"\d+", _num_es, s)
    s = sin_acentos(s.lower())
    s = re.sub(r"[^a-zñ0-9\s]", " ", s)
    return s.split()


def silabas_es(texto):
    total = 0
    for w in normaliza(texto):
        grupos = re.findall(r"[aeiouy]+", w)
        n = 0
        for g in grupos:
            # hiato de dos vocales fuertes (ae, eo, oa...) cuenta como 2
            n += 1 + sum(1 for a, b in zip(g, g[1:]) if a in "aeo" and b in "aeo")
        total += max(n, 1)
    return total


def palabras_guion(texto):
    """Palabras del guion con la puntuación que las sigue."""
    out = []
    for m in re.finditer(r"([\wáéíóúüñÁÉÍÓÚÜÑ]+)([^\wáéíóúüñÁÉÍÓÚÜÑ]*)", texto):
        out.append((m.group(1), m.group(2).strip()))
    return out


# ----------------------------------------------------------------------------- métricas
def dnsmos(y16):
    import onnxruntime as ort

    sess = ort.InferenceSession(os.path.join(MODELS, "sig_bak_ovr.onnx"))
    p808 = ort.InferenceSession(os.path.join(MODELS, "model_v8.onnx"))
    fs, L = 16000, 9.01
    aud = y16.copy()
    while len(aud) < int(L * fs):
        aud = np.append(aud, aud)
    hops = int(np.floor(len(aud) / fs) - L) + 1
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    S, B, O, P = [], [], [], []
    for i in range(hops):
        seg = aud[i * fs: int((i + L) * fs)]
        if len(seg) < int(L * fs):
            continue
        s, b, o = sess.run(None, {"input_1": seg.astype("float32")[None, :]})[0][0]
        mel = librosa.feature.melspectrogram(y=seg[:-160], sr=fs, n_fft=321, hop_length=160, n_mels=120)
        mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        P.append(p808.run(None, {"input_1": mel.T.astype("float32")[None, :, :]})[0][0][0])
        S.append(p_sig(s)); B.append(p_bak(b)); O.append(p_ovr(o))
    return dict(dnsmos_sig=float(np.mean(S)), dnsmos_bak=float(np.mean(B)),
                dnsmos_ovrl=float(np.mean(O)), dnsmos_p808=float(np.mean(P)),
                dnsmos_sig_min=float(np.min(S)), dnsmos_ovrl_min=float(np.min(O)))


_NISQA = None


def nisqa_tts(y, sr):
    global _NISQA
    import torch
    sys.path.insert(0, MODELS)
    import NISQA_lib as NL

    if _NISQA is None:
        ck = torch.load(os.path.join(MODELS, "nisqa_tts.tar"), map_location="cpu", weights_only=True)
        a = ck["args"]
        keys = ["ms_seg_length", "ms_n_mels", "cnn_model", "cnn_c_out_1", "cnn_c_out_2", "cnn_c_out_3",
                "cnn_kernel_size", "cnn_dropout", "cnn_pool_1", "cnn_pool_2", "cnn_pool_3", "cnn_fc_out_h",
                "td", "td_sa_d_model", "td_sa_nhead", "td_sa_pos_enc", "td_sa_num_layers", "td_sa_h",
                "td_sa_dropout", "td_lstm_h", "td_lstm_num_layers", "td_lstm_dropout", "td_lstm_bidirectional",
                "td_2", "td_2_sa_d_model", "td_2_sa_nhead", "td_2_sa_pos_enc", "td_2_sa_num_layers", "td_2_sa_h",
                "td_2_sa_dropout", "td_2_lstm_h", "td_2_lstm_num_layers", "td_2_lstm_dropout",
                "td_2_lstm_bidirectional", "pool", "pool_att_h", "pool_att_dropout"]
        m = NL.NISQA(**{k: a[k] for k in keys})
        m.load_state_dict(ck["model_state_dict"], strict=True)
        m.eval()
        _NISQA = (m, a)
    m, a = _NISQA

    def score(seg):
        S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=a["ms_n_fft"], hop_length=int(sr * a["ms_hop_length"]),
                                           win_length=int(sr * a["ms_win_length"]), window="hann", center=True,
                                           pad_mode="reflect", power=1.0, n_mels=a["ms_n_mels"], fmin=0.0,
                                           fmax=a["ms_fmax"], htk=False, norm="slaney")
        spec = librosa.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
        x, n = NL.segment_specs("seg", spec, a["ms_seg_length"], a["ms_seg_hop_length"], None)
        with torch.no_grad():
            return float(m(x.unsqueeze(0).float(), torch.tensor([int(n)]))[0][0])

    full = score(y)
    win, hop = int(6 * sr), int(3 * sr)
    vent = []
    for st in range(0, max(1, len(y) - win + 1), hop):
        vent.append((st / sr, score(y[st: st + win])))
    peor = min(vent, key=lambda t: t[1])
    return dict(nisqa_tts_total=full, nisqa_tts_media=float(np.mean([v for _, v in vent])),
                nisqa_tts_peor_ventana=peor[1], nisqa_tts_peor_en_s=peor[0],
                nisqa_ventanas=[(round(t, 1), round(v, 2)) for t, v in vent])


def prosodia(path):
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=350)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    st = 12 * np.log2(f0 / np.median(f0))
    pp = call(snd, "To PointProcess (periodic, cc)", 60, 350)
    jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(snd.to_harmonicity_cc(0.01, 60, 0.1, 1.0), "Get mean", 0, 0)
    # picos de tono exagerados (>+8 st sobre la mediana) y "vocal fry" (<-7 st)
    picos = float(np.mean(st > 8) * 100)
    fry = float(np.mean(st < -7) * 100)
    # saltos de volumen: sonoridad a corto plazo (400 ms) en tramos con voz
    y = snd.values[0]
    sr = int(snd.sampling_frequency)
    w = int(0.4 * sr)
    rms = np.array([np.sqrt(np.mean(y[i:i + w] ** 2)) for i in range(0, max(1, len(y) - w), int(0.1 * sr))])
    db = 20 * np.log10(rms + 1e-9)
    voz = db[db > db.max() - 30]
    salto = float(voz.max() - np.median(voz)) if len(voz) else 0.0
    return dict(f0_picos_pct=picos, f0_fry_pct=fry, salto_volumen_db=salto,
                f0_mediana_hz=float(np.median(f0)), f0_std_st=float(np.std(st)),
                f0_rango_st=float(np.percentile(st, 95) - np.percentile(st, 5)),
                jitter_pct=float(jit), shimmer_pct=float(shim), hnr_db=float(hnr))


def tecnica(y, sr):
    import pyloudnorm as pyln

    peak = float(np.max(np.abs(y)))
    clip = int(np.sum(np.abs(y) >= 0.999))
    lufs = float(pyln.Meter(sr).integrated_loudness(y))
    # clics: transitorios agudos muy por encima de la envolvente local
    from scipy.signal import butter, sosfilt
    sos = butter(4, 7000, "hp", fs=sr, output="sos") if sr > 16000 else None
    hp = sosfilt(sos, y) if sos is not None else y
    env = np.sqrt(np.convolve(hp ** 2, np.ones(int(0.02 * sr)) / int(0.02 * sr), mode="same")) + 1e-6
    ratio = np.abs(hp) / env
    cand = np.where((ratio > 14) & (np.abs(hp) > 0.02))[0]
    clics = 0
    last = -sr
    for c in cand:
        if c - last > 0.05 * sr:
            clics += 1
        last = c
    # silencios inicio/fin
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=int(0.02 * sr), hop_length=int(0.01 * sr))[0], ref=np.max)
    act = np.where(db > -40)[0]
    ini = act[0] * 0.01 if len(act) else 0
    fin = (len(db) - act[-1]) * 0.01 if len(act) else 0
    piso = float(np.percentile(db, 5))
    return dict(duracion_s=len(y) / sr, pico_dbfs=20 * math.log10(peak + 1e-9), clipping_muestras=clip,
                lufs=lufs, clics=clics, silencio_inicio_s=float(ini), silencio_final_s=float(fin),
                piso_ruido_db_rel=piso)


def silencios(y, sr, minimo=0.18):
    hop = int(0.01 * sr)
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=int(0.03 * sr), hop_length=hop)[0], ref=np.max)
    sil = db < -38
    out, i = [], 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if (j - i) * 0.01 >= minimo:
                out.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return out


_WH = None
ALUCINACIONES = {"gracias", "muchas", "por", "ver", "el", "video", "suscribete", "subtitulos", "realizados", "la",
                 "comunidad", "de", "amara", "org", "y", "hasta", "proxima", "adios"}


def transcribe(path):
    global _WH
    y16, _ = librosa.load(path, sr=16000, mono=True)
    from faster_whisper import WhisperModel

    if _WH is None:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _WH = WhisperModel(WHISPER, device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8",
                           cpu_threads=os.cpu_count() or 4)
    segs, _ = _WH.transcribe(y16.astype(np.float32), language="es", beam_size=5, word_timestamps=True, vad_filter=False,
                             condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in s.words:
            words.append((w.word.strip(), w.start, w.end))
    return words


def texto_y_ritmo(y, sr, path, guion):
    words = transcribe(path)
    hyp = " ".join(w for w, _, _ in words)
    ref_n, hyp_n = normaliza(guion), normaliza(hyp)
    sm = SequenceMatcher(a=ref_n, b=hyp_n, autojunk=False)
    errores, errores_idx, dist = [], [], 0
    ops = sm.get_opcodes()
    for op, a0, a1, b0, b1 in ops:
        if op != "equal":
            # Whisper a veces "alucina" una despedida en el silencio final ("gracias", "suscríbete"...)
            if op == "insert" and a0 == len(ref_n) and set(hyp_n[b0:b1]) <= ALUCINACIONES:
                continue
            dist += max(a1 - a0, b1 - b0)
            errores.append(f"{op}: guion='{' '.join(ref_n[a0:a1])}' audio='{' '.join(hyp_n[b0:b1])}'")
            errores_idx.append((a0, a1))
    wer = dist / max(len(ref_n), 1)

    # Alinear palabras de Whisper con las del guion para saber qué puntuación sigue a cada una
    # Se alinean por tokens normalizados: Whisper escribe "2020," como una sola palabra que equivale a
    # "dos mil veinte," en el guion. La puntuación de la palabra de Whisper es la de su ÚLTIMO token
    # (si solo se usara el primero, la pausa tras "2020," se atribuiría a "dos" y saldría como pausa rara).
    gw = palabras_guion(guion)
    a_tok, a_src = [], []  # tokens del guion -> (índice de palabra del guion, ¿es su último token?)
    for gi, (g, _) in enumerate(gw):
        ts = normaliza(g) or [""]
        for j, t in enumerate(ts):
            a_tok.append(t)
            a_src.append((gi, j == len(ts) - 1))
    b_tok, b_src = [], []  # tokens de Whisper -> índice de palabra de Whisper
    for wi, (w, _, _) in enumerate(words):
        for t in normaliza(w) or [""]:
            b_tok.append(t)
            b_src.append(wi)
    m2 = SequenceMatcher(a=a_tok, b=b_tok, autojunk=False)
    punct_de_hyp = {}
    for blk in m2.get_matching_blocks():
        for k in range(blk.size):
            gi, ultimo = a_src[blk.a + k]
            punct_de_hyp[b_src[blk.b + k]] = (gw[gi][0], gw[gi][1] if ultimo else "")  # gana el último token

    sil = silencios(y, sr)
    dur = len(y) / sr
    pausas, raras = [], 0
    for (s0, s1) in sil:
        if s0 < 0.05 or s1 > dur - 0.05:
            continue  # silencios de borde
        # palabra que termina justo antes del silencio
        idx = max([i for i, (_, a, b) in enumerate(words) if b <= s0 + 0.25] or [-1])
        if idx < 0:
            continue
        if pausas and pausas[-1].get("_idx") == idx:
            pausas[-1]["dur"] = round(pausas[-1]["dur"] + (s1 - s0), 2)
            continue
        palabra, punt = punct_de_hyp.get(idx, (words[idx][0], None))
        ok = bool(punt)
        d = s1 - s0
        if not ok and d >= 0.30:
            raras += 1
        pausas.append(dict(tras=palabra, punt=punt or "", dur=round(d, 2), en_s=round(s0, 2), rara=(not ok and d >= 0.30), _idx=idx))

    # pausas esperadas en puntos/interrogaciones/puntos suspensivos
    esperadas = [g for g, p in gw if p and any(c in p for c in ".?!…")]
    con_pausa = {p["tras"].lower() for p in pausas if p["punt"]}
    faltan = [g for g in esperadas[:-1] if g.lower() not in con_pausa]

    habla = dur - sum(p["dur"] for p in pausas)
    dp = [p["dur"] for p in pausas if p["punt"]]
    return dict(pausa_max_s=max(dp or [0]), pausa_media_s=float(np.mean(dp or [0])), transcripcion=hyp, wer=wer,
                errores=errores, errores_idx=errores_idx,
                silabas_por_seg=silabas_es(guion) / max(habla, 1e-3),
                pausas=pausas, pausas_raras=raras, pausas_puntuacion_faltantes=len(faltan),
                pausas_faltan_tras=faltan)


def espectrograma(y, sr, png, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import parselmouth

    fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax[0], cmap="magma", vmin=-90)
    ax[0].set_title(titulo)
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=350)
    f = p.selected_array["frequency"]; f[f == 0] = np.nan
    ax[1].plot(p.xs(), f, ".", ms=2)
    ax[1].set_ylabel("F0 (Hz)")
    t = np.arange(len(y)) / sr
    ax2 = ax[1].twinx(); ax2.plot(t[::50], y[::50], alpha=0.25, color="gray")
    plt.tight_layout(); plt.savefig(png, dpi=80); plt.close(fig)


def veredicto(r):
    fallos = []
    for k, (op, v) in UMBRALES.items():
        x = r.get(k)
        if x is None:
            continue
        ok = (x >= v) if op == ">=" else (x <= v) if op == "<=" else (v[0] <= x <= v[1])
        if not ok:
            fallos.append(f"{k}={x:.3f} (objetivo {op} {v})")
    return fallos


def evaluar(path, guion, png=None, nisqa=True, dns=True, asr=True):
    import librosa.display  # noqa: F401
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    r = {"archivo": os.path.basename(path)}
    r.update(tecnica(y, sr))
    r.update(prosodia(path))
    if dns:
        r.update(dnsmos(librosa.resample(y, orig_sr=sr, target_sr=16000)))
    if nisqa:
        r.update(nisqa_tts(y, sr))
    if asr:
        r.update(texto_y_ritmo(y, sr, path, guion))
    if png:
        espectrograma(y, sr, png, os.path.basename(path))
    r["fallos"] = veredicto(r)
    r["aprobado"] = not r["fallos"]
    return r


def resumen(r):
    k = ["nisqa_tts_media", "nisqa_tts_peor_ventana", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808",
         "wer", "f0_mediana_hz", "f0_std_st", "f0_rango_st", "f0_picos_pct", "f0_fry_pct", "salto_volumen_db", "jitter_pct", "shimmer_pct", "hnr_db",
         "silabas_por_seg", "pausa_media_s", "pausa_max_s", "pausas_raras", "pausas_puntuacion_faltantes", "clics", "pico_dbfs", "lufs", "duracion_s"]
    return " | ".join(f"{x}={r[x]:.2f}" if isinstance(r.get(x), float) else f"{x}={r.get(x)}" for x in k if x in r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("guion")
    ap.add_argument("--png")
    ap.add_argument("--json")
    a = ap.parse_args()
    g = open(a.guion, encoding="utf-8").read().strip()
    res = evaluar(a.audio, g, a.png)
    print(resumen(res))
    print("Transcripción:", res.get("transcripcion"))
    for e in res.get("errores", []):
        print("  ERROR TEXTO:", e)
    for p in res.get("pausas", []):
        print(f"  pausa {p['dur']:.2f}s tras '{p['tras']}{p['punt']}' @{p['en_s']}s{'  <-- RARA' if p['rara'] else ''}")
    print("Fallos:", res["fallos"] or "ninguno")
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1, default=float)
