from scipy.ndimage.morphology import binary_dilation
from codificador.params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct


#Un bloque de sentencias 
try:
    import webrtcvad  # Elimina la deteccion de ruidos y clasifica una pieza de datos de audio haciendo un reconocimiento de voz 
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.") 
    webrtcvad=None

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
   Aplica las operaciones de preprocesamiento utilizadas para entrenar el codificador de altavoz a una forma de onda
    ya sea en disco o en memoria. La forma de onda se volverá a muestrear para que coincida con los hiperparámetros de datos.

    :param fpath_or_wav: ya sea una ruta de archivo a un archivo de audio (se admiten muchas extensiones, no
    solo .wav), ya sea la forma de onda como una matriz numpy de flotadores.
    :param source_sr: si pasa una forma de onda de audio, la frecuencia de muestreo de la forma de onda anterior
    preprocesamiento Después del preprocesamiento, la frecuencia de muestreo de la forma de onda coincidirá con los datos
    hiperparámetros. Si pasa una ruta de archivo, la tasa de muestreo se detectará automáticamente y
    este argumento será ignorado.
    """
    # Cargue el wav desde el disco si es necesario
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Remuestrear la wav si es necesario
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Aplicar el preprocesamiento: normalizar el volumen y acortar los silencios largos 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav):
    """
   Deriva un espectrograma de mel listo para ser utilizado por el codificador a partir de una forma de onda de audio preprocesada.
    Nota: este no es un espectrograma.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav):
    """
   Garantiza que los segmentos sin voz en la forma de onda no permanezcan más de un
    umbral determinado por los parámetros VAD en params.py.

    :param wav: la forma de onda sin procesar como una matriz numpy de flotadores
    :return: la misma forma de onda con los silencios recortados (longitud <= longitud de onda original)
    """
    # Calcula el tamaño de la ventana de detección de voz
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    #  Recorte el final del audio para tener un múltiplo del tamaño de la ventana
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convierta la forma de onda flotante a PCM mono de 16 bits
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Realizar detección de activación por voz
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Suaviza la deteccion de voz con un medio movil
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilatar las regiones sonoras
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))
