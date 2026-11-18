---
name: narrar-guion
description: Convierte un guion en una locución de narrador natural (voz fija del canal Modo Paranoia) con control de calidad automático, y entrega MP3 + WAV con un informe. Úsala cuando el usuario pegue o adjunte un guion y pida narración, voz, locución, voiceover, audio o "pásalo a voz", aunque no nombre la skill.
---

# Narrar un guion con la voz del canal

El kit está en la carpeta `voz/` (en el repositorio, o dentro del zip `kit_narracion_modo_paranoia.zip`).
Hace en un solo comando lo que un estudio haría a mano: divide el guion en fragmentos, graba varias tomas
de cada uno con la voz fijada, las evalúa, repite las que fallen, elige la mejor combinación, ensambla con
pausas naturales, masteriza sin degradar la voz y verifica el resultado completo.

## Pasos

1. **Prepara el entorno (solo la primera vez en cada máquina).**
   - Si no existe `voz/.entorno.json`, ejecuta `bash voz/instalar.sh` en segundo plano (5-10 min, ~6 GB).
   - Necesita red hacia `pypi.org`, `huggingface.co`, `us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co` y
     `raw.githubusercontent.com`. Si una descarga de Hugging Face falla con 403/conexión rechazada, el entorno
     no permite esos hosts: el usuario tiene un entorno de nube llamado **"wao"** que ya los permite (pídele que
     abra la sesión con ese entorno), o propón añadir los hosts con las herramientas de entorno.
   - Requiere `ffmpeg` (el instalador lo instala si falta).

2. **Guarda el guion** tal cual en un `.txt` (UTF-8). No reescribas el texto del autor sin preguntar.
   El script ya convierte números, porcentajes, horas y `$` a palabras. Avisa al usuario (sin cambiarlo) si ves:
   acotaciones entre corchetes o paréntesis (se leerían en voz alta), emojis, URLs, siglas raras o abreviaturas.
   Consejos de escritura que sí puedes sugerir: la puntuación marca las pausas; `...` da suspense;
   una línea en blanco entre párrafos = pausa de 1 s; frases cortas suenan más naturales.

3. **Narra** (en segundo plano; puede tardar):
   ```bash
   python3 voz/narrar.py guion.txt salida_NOMBRE/            # 6 tomas por fragmento: máxima calidad
   python3 voz/narrar.py guion.txt salida_NOMBRE/ --tomas 3  # ~2x más rápido
   ```
   Tiempo orientativo en CPU de 4 núcleos (medido): ~30-40 min por minuto de narración con 6 tomas y
   ~15-25 min con `--tomas 3`, más ~2 min de carga (con GPU, muchísimo menos). Para guiones de más de
   2-3 minutos, propone `--tomas 3` o avisa del tiempo antes de empezar.
   El log muestra cada fragmento, cuántas tomas aprobaron y si se repitió alguno.

4. **Revisa el resultado**: `salida_NOMBRE/informe.txt` dice `APROBADO` o `CON AVISOS` y las métricas.
   Mira también `salida_NOMBRE/espectrograma.png` (busca clics como líneas verticales aisladas, ruido entre
   frases o saltos bruscos de tono).
   - Si sale `CON AVISOS`, repite en otra carpeta con otra semilla y más tomas/rondas
     (`--semilla 1 --tomas 8 --rondas 6`). Sin cambiar `--semilla` saldrían las mismas tomas. Avisa del tiempo.

5. **Entrega**: envía `salida_NOMBRE/narracion.mp3` (para escuchar) y `narracion.wav` (48 kHz/24 bit, para el
   editor), y resume en 3-5 líneas: duración, si aprobó, naturalidad (NISQA), palabras correctas (WER) y
   cualquier aviso. Di claramente que la evaluación es por mediciones: tú no puedes escuchar el audio, y su oído
   es la prueba final. La voz queda entre −16 y −20 LUFS (sin compresión): se ajusta en el editor al mezclar con música.

## Qué significa "aprobado" (umbrales calibrados con voz humana real)

| Métrica | Umbral | Qué evita |
|---|---|---|
| WER (Whisper) | 0 % | palabras omitidas, cambiadas o inventadas |
| NISQA-TTS media / peor ventana de 6 s | ≥ 4,0 / ≥ 3,6 | voz robótica o tramos raros |
| DNSMOS SIG / OVRL | ≥ 3,4 / ≥ 3,2 | distorsión |
| HNR / jitter | ≥ 10 dB / ≤ 3,0 % | voz rasposa o "sucia" |
| Variación de F0 | ≥ 2,5 semitonos | monotonía |
| Picos de F0 / vocal fry | ≤ 4 % / ≤ 3 % | entonación exagerada, carraspeo |
| Pausas raras / clics / clipping | 0 | cortes y defectos técnicos |

## No hagas

- No cambies la voz (`voz/voz_fijada/`) salvo que el usuario lo pida: es la identidad del canal. Para una voz
  nueva hay que repetir el casting (ver `voz/README.md`, modo `design` de `generar.py`).
- No añadas compresor, limitador, de-esser ni ecualización: se probaron y todos degradaban la voz.
- No digas que el audio "suena" de tal manera como si lo hubieras escuchado.
