
## Mel-filterbank
mel_window_length = 25  # En milisegundos
mel_window_step = 10    # En milisegundos
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Número de cuadros de espectrograma en un enunciado parcial
partials_n_frames = 160     # 1600 ms
# Número de cuadros de espectrograma en la inferencia
inference_n_frames = 80     #  800 ms


## Detección de activación de voz
# Tamaño de ventana del VAD. Debe ser 10, 20 o 30 milisegundos.
# Esto establece la granularidad del VAD. No debería ser necesario cambiarlo.
vad_window_length = 30  # En milisegundos
# Número de cuadros para promediar juntos cuando se realiza el suavizado de promedio móvil.
# Cuanto mayor sea este valor, mayores deben ser las variaciones del VAD para que no se suavicen.
vad_moving_average_width = 8
# Número máximo de fotogramas silenciosos consecutivos que puede tener un segmento.
vad_max_silence_length = 6


## Normalización del volumen de audio
audio_norm_target_dBFS = -30

