import platform
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthesizer.hparams import hparams_debug_string
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils import data_parallel_workaround
from synthesizer.utils.symbols import symbols


def run_synthesis(in_dir: Path, out_dir: Path, syn_model_fpath: Path, hparams):
    # Esto genera mels alineados con la realidad del suelo para el entrenamiento del codificador de voz
    synth_dir = out_dir / "mels_gta"
    synth_dir.mkdir(exist_ok=True, parents=True)
    print(hparams_debug_string())

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if hparams.synthesis_batch_size % torch.cuda.device_count() != 0:
            raise ValueError("`hparams.synthesis_batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Synthesizer using device:", device)

    # Crear una instancia del modelo Tacotron
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=0., # Use zero dropout for gta mels
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Carga los weights
    print("\nLoading weights at %s" % syn_model_fpath)
    model.load(syn_model_fpath)
    print("Tacotron weights loaded from step %d" % model.step)

    # Sintetice usando el mismo factor de reducción que el modelo actualmente entrenado
    r = np.int32(model.r)

    # Establecer el modelo en modo de evaluación (deshabilitar gradiente y zoneout)
    model.eval()

    # Initialize the dataset
    metadata_fpath = in_dir.joinpath("train.txt")
    mel_dir = in_dir.joinpath("mels")
    embed_dir = in_dir.joinpath("embeds")

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)
    collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
    data_loader = DataLoader(dataset, hparams.synthesis_batch_size, collate_fn=collate_fn, num_workers=2)

    # Generate GTA mels
    meta_out_fpath = out_dir / "synthesized.txt"
    with meta_out_fpath.open("w") as file:
        for i, (texts, mels, embeds, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            texts, mels, embeds = texts.to(device), mels.to(device), embeds.to(device)

            # Paralelice el modelo en GPUS usando una solución alternativa debido a un error de python
            if device.type == "cuda" and torch.cuda.device_count() > 1:
                _, mels_out, _ = data_parallel_workaround(model, texts, mels, embeds)
            else:
                _, mels_out, _, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                # Nota: los archivos de espectrograma mel de salida y los de destino tienen los mismos nombres, solo carpetas diferentes
                mel_filename = Path(synth_dir).joinpath(dataset.metadata[k][1])
                mel_out = mels_out[j].detach().cpu().numpy().T

                # Use la longitud de la verdad de tierra mel para eliminar el relleno de las mels generadas
                mel_out = mel_out[:int(dataset.metadata[k][4])]

                # Escribir del spectrogram a disk
                np.save(mel_filename, mel_out, allow_pickle=False)

                # Escribir metadatos en el archivo sintetizado
                file.write("|".join(dataset.metadata[k]))
