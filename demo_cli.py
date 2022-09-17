import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from codificador import inference as encoder
from codificador.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Oculte las GPU de Pytorch para forzar el procesamiento de la CPU
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Imprime alguna información del entorno (para fines de depuración)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Cargue los modelos uno por uno.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    ## Haz una prueba
    print("Testing your configuration with small inputs.")
     # Reenvía una forma de onda de audio de ceros que dura 1 segundo. Observe cómo podemos obtener el codificador
     # tasa de muestreo, que puede diferir.
     # Si no está familiarizado con el audio digital, sepa que está codificado como una matriz de flotantes
     # (o, a veces, números enteros, pero en su mayoría flotantes en estos proyectos) que van desde -1 a 1.
     # La frecuencia de muestreo es el número de valores (muestras) registrados por segundo, se establece en
     # 16000 para el codificador. Crear una matriz de longitud <tasa de muestreo> siempre corresponderá
     # a un audio de 1 segundo.
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))

     # Crea una incrustación ficticia. Normalmente usaría la incrustación que encoder.embed_utterance
     # regresa, pero aquí vamos a hacer uno solo para mostrar que es
     # posible.
    embed = np.random.rand(speaker_embedding_size)
    # Las incrustaciones están normalizadas en L2 (esto no es importante aquí, pero si quiere hacer su propio
     # incrustaciones será).
    embed /= np.linalg.norm(embed)
    # El sintetizador puede manejar múltiples entradas con procesamiento por lotes. Vamos a crear otra incrustación para
     # ilustrar eso
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)

     # El vocoder sintetiza una forma de onda a la vez, pero es más eficiente para formas largas. Nosotros
     # puede concatenar los espectrogramas mel en uno solo.
    mel = np.concatenate(mels, axis=1)
    # El vocoder puede tomar una función de devolución de llamada para mostrar la generación. Más sobre eso más adelante. Para
     # ahora simplemente lo ocultaremos así:
    no_action = lambda *args: None
    print("\tTesting the vocoder...")
    # For the sake of making this test short, we'll pass a short target length. The target length
    # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
    # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
    # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
    # that has a detrimental effect on the quality of the audio. The default parameters are
    # recommended in general.
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

    print("All test passed! You can now synthesize speech.\n\n")


    ## Generación de voz interactiva
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")

    print("Interactive generation loop")
    num_generated = 0
    while True:
        try:
            # # Obtenga la ruta del archivo de audio de referencia
            message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                      "wav, m4a, flac, ...):\n"
            in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

            ## Calculando la incrustación
             # Primero, cargamos el wav usando la función que proporciona el codificador del altavoz. Esto es
             # importante: hay preprocesamiento que se debe aplicar.

           # Los siguientes dos métodos son equivalentes:
             # - Cargar directamente desde la ruta del archivo:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - # - Si el wav ya está cargado:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")

      # Luego derivamos la incrustación. Hay muchas funciones y parámetros que el
             # interfaces de codificador de altavoz. Estos son principalmente para la investigación en profundidad. normalmente
             # solo use esta función (con sus parámetros predeterminados):
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")


            ## Generating the spectrogram
            text = input("Write a sentence (+-20 words) to be synthesized:\n")

            # Si se especifica la seed, reinicie la semilla de la antorcha y fuerce la recarga del sintetizador
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

           # El sintetizador funciona por lotes, por lo que debe colocar sus datos en una lista o matriz numérica
            texts = [text]
            embeds = [embed]
             # Si sabe cuáles son las alineaciones de la capa de atención, puede recuperarlas aquí
             # pasando return_alignments=Verdadero
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")


            ## Generando la forma de onda
            print("Synthesizing the waveform:")

           # Si se especifica el seed, reinicie la semilla de la antorcha y vuelva a cargar el vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)
             # Sintetizar la forma de onda es bastante sencillo. Recuerda que cuanto más tiempo
             # espectrograma, más eficiente en el tiempo es el vocoder.
            generated_wav = vocoder.infer_waveform(spec)


            ## Post-generación
             # Hay un error con el dispositivo de sonido que hace que el audio se corte un segundo antes, así que
             # rellenarlo.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Recorte el exceso de silencios para compensar las lagunas en los espectrogramas (problema #53)
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Reproducir el audio (sin bloqueo)
            if not args.no_sound:
                import sounddevice as sd
                try:
                    sd.stop()
                    sd.play(generated_wav, synthesizer.sample_rate)
                except sd.PortAudioError as e:
                    print("\nCaught exception: %s" % repr(e))
                    print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
                except:
                    raise

           # Guardarlo en el disco
            filename = "demo_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)


        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
