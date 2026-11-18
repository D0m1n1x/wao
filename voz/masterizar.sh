#!/usr/bin/env bash
# Masterización de locución para YouTube: limpia, transparente, sin colorear la voz.
#   - remuestreo 24 kHz -> 48 kHz con soxr (alta calidad)
#   - paso-altos 70 Hz (quita retumbe), de-esser suave
#   - compresión 2:1 muy ligera para igualar volumen sin "bombeo"
#   - normalización a -16 LUFS integrados, pico real <= -1.5 dBTP (dos pasadas)
#   - 0,3 s de silencio al inicio y 0,6 s al final
# Uso: ./masterizar.sh entrada.wav salida_base   -> salida_base.wav (48k/24bit) y salida_base.mp3 (320k)
set -euo pipefail
in="$1"; out="$2"
pre="highpass=f=70:poles=2,deesser=i=0.25:m=0.5:f=0.5,acompressor=threshold=-20dB:ratio=2:attack=15:release=180:makeup=1,silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.05,areverse,silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.05,areverse,adelay=300:all=1,apad=pad_dur=0.6"
stats=$(ffmpeg -hide_banner -nostats -i "$in" -af "aresample=48000:resampler=soxr:precision=28,$pre,loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json" -f null - 2>&1 | sed -n '/^{/,/^}/p')
mi=$(echo "$stats" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['input_i'])")
mtp=$(echo "$stats" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['input_tp'])")
mlra=$(echo "$stats" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['input_lra'])")
mth=$(echo "$stats" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['input_thresh'])")
off=$(echo "$stats" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['target_offset'])")
ffmpeg -hide_banner -loglevel error -y -i "$in" -af "aresample=48000:resampler=soxr:precision=28,$pre,loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=$mi:measured_TP=$mtp:measured_LRA=$mlra:measured_thresh=$mth:offset=$off:linear=true,aresample=48000:resampler=soxr" -ar 48000 -c:a pcm_s24le "$out.wav"
ffmpeg -hide_banner -loglevel error -y -i "$out.wav" -c:a libmp3lame -b:a 320k "$out.mp3"
echo "OK $out.wav $out.mp3"
