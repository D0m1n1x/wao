# Cómo narrar tus guiones desde otro chat

Necesitas una sesión de **Claude Code en la web** (claude.ai/code). El chat normal de claude.ai no sirve,
porque el proceso descarga ~6 GB de modelos y necesita varios minutos de CPU.

En la sesión nueva, elige siempre el **entorno de nube "wao"**: está en el menú de entorno de la barra de título
de la sesión. Ese entorno ya permite descargar los modelos de Hugging Face. Con el entorno "Default", la
instalación fallará.

---

## Opción A: desde el repositorio (cuando GitHub tenga acceso)

1. Arregla el acceso una vez: reconecta GitHub en https://claude.ai/connect-github e instala la app de Claude
   en el repositorio `D0m1n1x/wao`. Después pide en esta sesión que suba los cambios.
2. Abre una sesión nueva en claude.ai/code con el repositorio **D0m1n1x/wao**, la rama
   **claude/perfeccionar-voz-natural-45w1pt** (o `main` si ya la fusionaste) y el entorno **wao**.
3. Pega este mensaje y debajo tu guion:

```
Narra este guion con la voz fija del canal usando la skill narrar-guion
(.claude/skills/narrar-guion/SKILL.md y la carpeta voz/). Entrégame el MP3, el WAV y el informe de calidad.

GUION:
<pega aquí el guion>
```

## Opción B: con el kit en zip (funciona ya, sin GitHub)

1. Abre una sesión nueva en claude.ai/code con el entorno **wao** (el repositorio da igual).
2. Adjunta `kit_narracion_modo_paranoia.zip` y pega:

```
Descomprime kit_narracion_modo_paranoia.zip en el directorio de trabajo. Lee
kit/.claude/skills/narrar-guion/SKILL.md y síguelo al pie de la letra (el kit está en kit/voz/).
Narra este guion con la voz fija del canal y entrégame el MP3, el WAV y el informe de calidad.

GUION:
<pega aquí el guion>
```

---

## Qué esperar

- **Primera vez en cada sesión**: la instalación tarda unos 5-10 minutos.
- **Tiempo de narración** (en CPU): unos 15-20 minutos por cada minuto de audio con la máxima calidad
  (6 tomas por fragmento). Con `--tomas 3` tarda la mitad.
- Recibirás `narracion.mp3`, `narracion.wav` (48 kHz/24 bit) e `informe.txt` con APROBADO o CON AVISOS.
- La voz queda a unos −20 LUFS. Súbela en el editor al mezclar con la música (YouTube normaliza a −14 LUFS).

## Para que suene mejor, al escribir el guion

- La puntuación manda: punto = pausa, `...` = pausa de suspense, línea en blanco = pausa de 1 s.
- Frases cortas suenan más naturales que frases largas con muchas comas.
- No pongas acotaciones como `[susurrando]` o `(pausa)`: se leerían en voz alta.
- Los números (`1968`, `3:17`, `45%`, `$2.500`) se convierten solos a palabras.
- Escribe las siglas como se pronuncian si no son muy conocidas ("la CIA" va bien; una sigla rara, mejor escribirla entera).
