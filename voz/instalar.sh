#!/usr/bin/env bash
# Instala todo lo necesario para narrar.py (una sola vez por máquina; tarda ~5-10 min y ocupa ~6 GB).
#   - Librerías de Python (PyPI)
#   - Modelo de voz Qwen3-TTS 1.7B Base (Hugging Face, Apache-2.0)
#   - Whisper large-v3-turbo para comprobar palabras (Hugging Face)
#   - DNSMOS (Microsoft) y NISQA-TTS desde sus repositorios oficiales de GitHub
# Requiere acceso de red a: pypi.org, huggingface.co (+ us.aws.cdn.hf.co, cas-bridge.xethub.hf.co),
# raw.githubusercontent.com. Uso: bash instalar.sh   (opcional: VOZ_HOME=/ruta bash instalar.sh)
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOZ_HOME="${VOZ_HOME:-$HOME/.cache/voz_modo_paranoia}"
mkdir -p "$VOZ_HOME/models" "$VOZ_HOME/hf"

command -v ffmpeg >/dev/null || { echo "Instalando ffmpeg..."; (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null; }

echo "[1/4] Librerías de Python..."
pip install -q --disable-pip-version-check --use-pep517 torch qwen-tts faster-whisper librosa praat-parselmouth pyloudnorm \
  onnxruntime num2words soundfile scipy matplotlib huggingface_hub 2>&1 | grep -v "Running pip as the 'root'" || true

echo "[2/4] DNSMOS y NISQA-TTS (repositorios oficiales)..."
M="$VOZ_HOME/models"
[ -s "$M/sig_bak_ovr.onnx" ] || curl -sSfL -o "$M/sig_bak_ovr.onnx" https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
[ -s "$M/model_v8.onnx" ]    || curl -sSfL -o "$M/model_v8.onnx"    https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx
[ -s "$M/nisqa_tts.tar" ]    || curl -sSfL -o "$M/nisqa_tts.tar"    https://raw.githubusercontent.com/gabrielmittag/NISQA/master/weights/nisqa_tts.tar
[ -s "$M/NISQA_lib.py" ]     || curl -sSfL -o "$M/NISQA_lib.py"     https://raw.githubusercontent.com/gabrielmittag/NISQA/master/nisqa/NISQA_lib.py

echo "[3/4] Modelos de Hugging Face (voz ~4,5 GB + Whisper ~1,6 GB)..."
cd /tmp
HF_HOME="$VOZ_HOME/hf" python3 - "$AQUI/.entorno.json" "$VOZ_HOME" <<'EOF'
import json, sys, time
from huggingface_hub import snapshot_download
rutas = {}
for clave, repo in [("QWEN_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                    ("WHISPER_MODEL", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")]:
    for intento in range(5):
        try:
            rutas[clave] = snapshot_download(repo)
            print("   OK", repo, flush=True)
            break
        except Exception as e:
            print(f"   reintento {intento + 1} {repo}: {str(e)[:150]}", flush=True)
            time.sleep(5 * (intento + 1))
    else:
        sys.exit(f"No se pudo descargar {repo}. ¿Está permitido huggingface.co (y us.aws.cdn.hf.co) en la red?")
rutas["VOZ_MODELS"] = sys.argv[2] + "/models"
rutas["HF_HOME"] = sys.argv[2] + "/hf"
json.dump(rutas, open(sys.argv[1], "w"), indent=1)
EOF

echo "[4/4] Comprobación..."
python3 -c "import qwen_tts, faster_whisper, parselmouth, onnxruntime, num2words; print('   librerías OK')" 2>&1 | grep -v -i "warning\|flash\|sox\|^\*\|^ *$\|sourceforge\|path variables\|If you do"
echo "Listo. Rutas guardadas en $AQUI/.entorno.json"
echo "Para narrar:  python3 $AQUI/narrar.py guion.txt salida/"
