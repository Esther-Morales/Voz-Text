import ast
import pprint

class HParams(object):
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)

    def parse(self, string):
        # Anula hparams de una cadena separada por comas de pares nombre=valor
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

hparams = HParams(
        ### Procesamiento de señal (usado tanto en sintetizador como en vocoder)
        sample_rate = 16000,
        n_fft = 800,
        num_mels = 80,
        hop_size = 200,                             # Tacotron utiliza un cambio de cuadro de 12,5 ms (establecido en sample_rate * 0,0125)
        win_size = 800,                             # Tacotron usa una longitud de cuadro de 50 ms (establecido en sample_rate * 0.050)
        fmin = 55,
        min_level_db = -100,
        ref_level_db = 20,
        max_abs_value = 4.,                         # El gradiente explota si es demasiado grande, la convergencia prematura si es demasiado pequeña.
        preemphasis = 0.97,                         # Coeficiente de filtro a usar si el énfasis previo es Verdadero
        preemphasize = True,

        ### Tacotron Text-to-Speech (TTS)
        tts_embed_dims = 512,                       # Dimensión de incrustación para las entradas de grafemas/fonemas
        tts_encoder_dims = 256,
        tts_decoder_dims = 128,
        tts_postnet_dims = 512,
        tts_encoder_K = 5,
        tts_lstm_dims = 1024,
        tts_postnet_K = 5,
        tts_num_highways = 4,
        tts_dropout = 0.5,
        tts_cleaner_names = ["english_cleaners"],
        tts_stop_threshold = -3.4,                  # Valor por debajo del cual finaliza la generación de audio.
                                                    # Por ejemplo, para un rango de [-4, 4], esto
                                                    # terminará la secuencia en el primer
                                                    # cuadro que tiene todos los valores < -3.4

        ### Entrenamiento Tacotron 
        tts_schedule = [(2,  1e-3,  20_000,  12),   # Programa de entrenamiento progresivo
                        (2,  5e-4,  40_000,  12),   # (r, lr, step, batch_size)
                        (2,  2e-4,  80_000,  12),   #
                        (2,  1e-4, 160_000,  12),   # r = reduction factor (# of mel frames
                        (2,  3e-5, 320_000,  12),   #     sintetizado para cada iteración del decodificador)
                        (2,  1e-5, 640_000,  12)],  # lr = learning rate

        tts_clip_grad_norm = 1.0,                   # recorta la norma de degradado para evitar explosiones; configúrelo en Ninguno si no es necesario
        tts_eval_interval = 500,                    # Número de pasos entre la evaluación del modelo (generación de la muestra)
                                                    # Establézcalo en -1 para generar después de completar la época, o 0 para deshabilitar

        tts_eval_num_samples = 1,                   # Hace que este número de muestras

        ### procesamiento de Datos 
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 16,                  # Para preprocesamiento e inferencia de vocoder.

        ###  Visualization de Mel y  Griffin-Lim
        signal_normalization = True,
        power = 1.5,
        griffin_lim_iters = 60,

        ### Opciones de procesamiento de audio
        fmax = 7600,                                # No exceder (sample_rate // 2)
        allow_clipping_in_normalization = True,     # se usa cuando signal_normalization = True
        clip_mels_length = True,                    # Si es verdadero, descarta muestras que excedan max_mel_frames
        use_lws = False,                            # "Recuperación rápida de la fase del espectrograma utilizando sumas ponderadas locales"
        symmetric_mels = True,                      # Sets mel range to [-max_abs_value, max_abs_value] if True,
                                                    #               and [0, max_abs_value] if False
        trim_silence = True,                        # Use with sample_rate of 16000 for best results

        ### SV2TTS
        speaker_embedding_size = 256,              # Dimensión para la incrustación del altavoz
        silence_min_duration_split = 0.4,           # Duración en segundos de un silencio para dividir un enunciado
        utterance_min_duration = 1.6,               # Duración en segundos por debajo de la cual se descartan los enunciados
        )

def hparams_debug_string():
    return str(hparams)
