# Voz de narrador natural para Modo Paranoia

**Resultado:** `final/narracion_modo_paranoia.mp3` (y `.wav` 48 kHz / 24 bit), 31,5 s.
Voz masculina medio-grave (~97 Hz), acento latinoamericano neutro, tono de narrador de misterio cercano.

## Guion

> Son las tres y diecisiete de la mañana. Te despierta la vibración del teléfono... pero no hay ninguna
> notificación. Lo desbloqueas, y en tu galería hay una foto nueva. Eres tú. Dormido. Tomada hace cinco
> minutos... desde dentro de tu habitación. ¿Paranoia? Tal vez. Pero lo que vas a escuchar hoy te va a hacer
> mirar tu teléfono de otra manera. Quédate hasta el final, porque te voy a enseñar a descubrir si alguien
> más tiene acceso a él. Ahora mismo.

## Métricas de la toma final (todas dentro de los umbrales)

| Qué mide | Métrica | Final | Umbral | Voz humana real* |
|---|---|---|---|---|
| Naturalidad | NISQA-TTS (MOS 1-5), media | **4,18** | ≥ 4,0 | 3,4 – 3,9 |
| Naturalidad del peor tramo | NISQA, peor ventana de 6 s | **3,94** | ≥ 3,6 | 2,7 – 3,3 |
| Distorsión de la señal | DNSMOS SIG | 3,68 | ≥ 3,4 | 3,7 |
| Calidad global percibida | DNSMOS P.808 | 4,33 | — | 3,9 – 4,2 |
| Palabras correctas | WER con Whisper large-v3-turbo | **0 %** | 0 % | — |
| Voz limpia (no rasposa) | HNR | **12,4 dB** | ≥ 10 dB | 10 – 11 dB |
| Aspereza | Jitter | 2,95 % | ≤ 3,0 % | 2,2 – 2,6 % |
| No monótona (no robótica) | Variación de F0 | 2,9 semitonos | ≥ 2,5 | 3,1 – 3,7 |
| Sin exageraciones | Picos de tono / vocal fry | 0,7 % / 0 % | ≤ 4 % / ≤ 3 % | hasta 4 % |
| Ritmo | Sílabas por segundo | 6,5 | 4,5 – 7,5 | ~7,8 en español leído** |
| Pausas | Pausas fuera de puntuación | 0 | 0 | — |
| Técnica | Clics / clipping / pico | 0 / 0 / −1 dBTP | 0 / 0 | — |

\* Medido con el mismo evaluador sobre grabaciones reales de LibriSpeech (narrador masculino).
\** Pellegrino et al. (2011), *A cross-language perspective on speech information rate*.

## Cómo se consiguió (y qué se descartó)

1. **Magnific/ElevenLabs**: no disponible (plan gratuito sin cupo de voz ese día).
2. **Motor**: Qwen3-TTS 1.7B (Apache-2.0, uso comercial permitido) en local, CPU.
3. **Evaluador objetivo** (`evaluar.py`): naturalidad (NISQA-TTS), distorsión (DNSMOS), exactitud del texto
   (Whisper), entonación, jitter/HNR, ritmo, pausas, clics, clipping. Umbrales calibrados con voz humana real.
4. **Rondas 1-2 (diseño de voz por descripción)**: cada semilla daba un narrador distinto (82-192 Hz) y las
   voces graves salían con *vocal fry* (rasposas). Descartado como método final.
5. **Casting** (12 voces leyendo un texto neutro) → ganó la voz **F** (`voz_fijada/`), la más natural y limpia.
6. **Voz fijada por clonación** (modelo Base + clip de referencia): todas las tomas son el mismo narrador.
7. **Punch-in**: el guion se dividió en 5 fragmentos; se generaron 6-30 tomas por fragmento, se evaluaron todas
   y se buscó la combinación con mejor naturalidad que cumpliera el jitter global (`optimizar_combinacion.py`).
   Se unieron con pausas controladas (0,6-0,9 s entre frases) y fundidos de 10 ms (`ensamblar.py`).
8. **Masterización**: se probaron compresor, de-esser, paso-altos y limitador; **todos empeoraban** la voz
   (NISQA 4,24 → 3,80 en el primer intento). La final es transparente: solo remuestreo a 48 kHz y ganancia
   lineal (`masterizar.sh`). Queda a −20 LUFS: súbela en el editor al mezclar con la música.

Limitación conocida: el modelo genera audio a 24 kHz, así que la voz no tiene contenido por encima de 12 kHz
(normal en TTS; en una mezcla con música no se nota).

## Reutilizar esta voz con otro guion

Requiere los modelos `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (generar), `mobiuslabsgmbh/faster-whisper-large-v3-turbo`
(evaluar) y los modelos de DNSMOS/NISQA (ver variables en `evaluar.py`).

1. Escribe los fragmentos (1-3 frases cada uno) y las pausas entre ellos en un `fragmentos.json`.
2. Genera tomas por fragmento con `generar.py` en modo `clone`, usando
   `voz_fijada/narrador_F_ref.wav` y su texto `voz_fijada/narrador_F_ref.txt` (ver `generar.py`).
3. `python3 elegir_fragmentos.py tomas/carpeta fragmentos.json tomas/ensamblado.wav`
4. `python3 optimizar_combinacion.py tomas/carpeta fragmentos.json 2.97 4`
5. `./masterizar.sh tomas/comboX.wav final/nombre` y vuelve a pasar `evaluar.py` al resultado.
