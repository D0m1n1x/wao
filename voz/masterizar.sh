#!/usr/bin/env bash
# Masterización TRANSPARENTE de la locución (verificada con evaluar.py: no cambia la calidad).
#   - remuestreo 24 kHz -> 48 kHz con soxr (precisión 28 bits)
#   - ganancia lineal para dejar el pico real en -1 dBTP (sin compresor, sin limitador, sin EQ)
#   - 0,3 s de silencio al inicio y 0,6 s al final (cómodo para editar)
# Se probaron compresor, de-esser, paso-altos y limitador: todos bajaban la naturalidad (NISQA)
# y la limpieza (HNR), así que se descartaron. Si necesitas más volumen, súbelo en el editor
# al mezclar con la música (YouTube normaliza la reproducción a -14 LUFS).
# Uso: ./masterizar.sh entrada.wav salida_base  -> salida_base.wav (48 kHz/24 bit) y salida_base.mp3 (320 kbps)
set -euo pipefail
in="$1"; out="$2"
tmp="$(mktemp --suffix=.wav)"
ffmpeg -hide_banner -loglevel error -y -i "$in" \
  -af "aresample=48000:resampler=soxr:precision=28,adelay=300:all=1,apad=pad_dur=0.6" -c:a pcm_f32le "$tmp"
python3 - "$tmp" "$out.wav" <<'EOF'
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
y, sr = sf.read(sys.argv[1])
tp = np.max(np.abs(resample_poly(y, 4, 1)))          # pico real aproximado (sobremuestreo x4)
g = 10 ** (-1.0 / 20) / tp
sf.write(sys.argv[2], y * g, sr, subtype="PCM_24")
print(f"ganancia lineal {20 * np.log10(g):+.2f} dB")
EOF
rm -f "$tmp"
ffmpeg -hide_banner -loglevel error -y -i "$out.wav" -c:a libmp3lame -b:a 320k "$out.mp3"
echo "OK $out.wav $out.mp3"
