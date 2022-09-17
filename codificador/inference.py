from codificador.params_data import *
from codificador.model import SpeakerEncoder
from codificador.audio import preprocess_wav   # Queremos exponer esta función desde aquí.
from matplotlib import cm
from codificador import audio
from pathlib import Path
import numpy as np
import torch

_model = None # tipo: codificador de altavoz
_device = None # tipo: antorcha.dispositivo


def load_model(weights_fpath: Path, device=None):
    """
    Carga el modelo en memoria. Si esta función no se llama explícitamente, se ejecutará en el
    primera llamada a embed_frames() con el archivo de pesos predeterminado.

    :param weights_fpath: la ruta para guardar los pesos del modelo.
    :param dispositivo: ya sea un torch device  o el nombre de un torch device (por ejemplo, "cpu", "cuda"). los
    el modelo se cargará y se ejecutará en este dispositivo. Sin embargo, las salidas siempre estarán en la CPU.
    Si no hay ninguno, se establecerá de forma predeterminada en su GPU si está disponible; de ​​lo contrario, su CPU.
    """
    # TODO: si la carga es lenta tiene mucho que ver con su pc 
   
    global _model, _device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    _model = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))


def is_loaded():
    return _model is not None


def embed_frames_batch(frames_batch):
    """
   Calcula las incorporaciones para un lote de espectrograma de mel.

    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed


def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
   Calcula dónde dividir una forma de onda de expresión y su correspondiente espectrograma mel para obtener
    expresiones parciales de <partial_utterance_n_frames> cada una. Tanto la forma de onda como la mel.
    se devuelven segmentos de espectrograma, para que cada forma de onda de expresión parcial corresponda a
    su espectrograma. Esta función asume que los parámetros del espectrograma de mel utilizados son los
    definido en params_data.py.

    The returned ranges may be indexing further than the length of the waveform. It is
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

    
    :param n_samples: el número de muestras en la forma de onda
    :param partial_utterance_n_frames: el número de fotogramas del espectrograma mel en cada parcial
    declaración
    :param min_pad_coverage: al llegar al último enunciado parcial, puede tener o no
    suficientes marcos. Si al menos <min_pad_coverage> de <partial_utterance_n_frames> están presentes,
    entonces se considerará el último enunciado parcial, como si rellenamos el audio. De lo contrario,
    se descartará, como si recortáramos el audio. Si no hay suficientes fotogramas para 1 parcial
    expresión, este parámetro se ignora para que la función siempre devuelva al menos 1 segmento.
    :param superposición: cuánto debe superponerse la expresión parcial. Si se establece en 0, el parcial
    los enunciados son completamente inconexos.
    :return: los cortes de forma de onda y los cortes de espectrograma de mel como listas de cortes de matriz. Índice
    respectivamente la forma de onda y el espectrograma mel con estos cortes para obtener el parcial
    declaraciones
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # calculas los  slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluar si se justifica o no el acolchado adicional
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Calcula una incrustación para una sola expresión.

    # TODO: manejar múltiples wavs para beneficiarse del procesamiento por lotes en GPU
    :param wav: una forma de onda de expresión preprocesada (ver audio.py) como una matriz numpy de float32
    :param using_partials: si es Verdadero, entonces el enunciado se divide en enunciados parciales de
    <partial_utterance_n_frames> fotogramas y la incrustación de la expresión se calcula a partir de sus
    media normalizada. Si es Falso, la expresión se calcula alimentando todo el
    espectrograma a la red.
    :param return_partials: si es True, las incrustaciones parciales también se devolverán junto con el
    rebanadas wav que corresponden a las incrustaciones parciales.
    :param kwargs: argumentos adicionales para compute_partial_splits()
    :return: la incrustación como una matriz numpy de float32 de forma (model_embedding_size,). Si
    <return_partials> es verdadero, las expresiones parciales como una matriz numpy de float32 de forma
    (n_partials, model_embedding_size) y los parciales wav como una lista de cortes también serán
    devuelto Si <using_partials> se establece simultáneamente en False, ambos valores serán None
    en cambio.
    """
    # Procesar el enunciado completo si no se usan parciales
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Calcular dónde dividir el enunciado en parciales y rellenar si es necesario
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Dividir el enunciado en parciales
    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    # Calcule la incrustación de la expresión a partir de las incrustaciones parciales
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def embed_speaker(wavs, **kwargs):
    raise NotImplemented()


def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)

    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
