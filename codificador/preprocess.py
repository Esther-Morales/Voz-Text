from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from codificador import audio
from codificador.config import librispeech_datasets, anglophone_nationalites
from codificador.params_data import *

# Las extenciones de audio que puede utilizar 
_AUDIO_EXTENSIONS = ("wav", "flac", "m4a", "mp3") # Este es el archivo preprocesado/aschivo principal

class DatasetLog:
    """
    Registra metadatos sobre el conjunto de datos en un archivo de texto.
    """
    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        from codificador import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()


def _init_preprocess_dataset(dataset_name, datasets_root, out_dir) -> (Path,DatasetLog): # solo se quita el Path para que deje de salir el error  se pueda grabar 
    dataset_root = datasets_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker(speaker_dir: Path, datasets_root: Path, out_dir: Path, skip_existing: bool):
    # Asigne un nombre al hablante que incluya su conjunto de datos
    speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)

    
    # Cree un directorio de salida con ese nombre, así como un archivo txt que contenga un
    # referencia a cada archivo fuente.
    speaker_out_dir = out_dir.joinpath(speaker_name)
    speaker_out_dir.mkdir(exist_ok=True)
    sources_fpath = speaker_out_dir.joinpath("_sources.txt")

    # Existe la posibilidad de que el preprocesamiento se haya interrumpido antes, verifique si
    # ya hay un archivo de fuentes.
    if sources_fpath.exists():
        try:
            with sources_fpath.open("r") as sources_file:
                existing_fnames = {line.split(",")[0] for line in sources_file}
        except:
            existing_fnames = {}
    else:
        existing_fnames = {}

    # Reúna todos los archivos de audio para ese altavoz de forma recursiva
    sources_file = sources_fpath.open("a" if skip_existing else "w")
    audio_durs = []
    for extension in _AUDIO_EXTENSIONS:
        for in_fpath in speaker_dir.glob("**/*.%s" % extension):
            # Comprobar si el archivo de salida de destino ya existe
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            # Cargar y preprocesar la forma de onda
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue

            # Crear el espectrograma mel, descartar los que son demasiado cortos
            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))
            audio_durs.append(len(wav) / sampling_rate)

    sources_file.close()

    return audio_durs


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger):
    print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))

    # Procesar los enunciados de cada hablante
    work_fn = partial(_preprocess_speaker, datasets_root=datasets_root, out_dir=out_dir, skip_existing=skip_existing)
    with Pool(4) as pool:
        tasks = pool.imap(work_fn, speaker_dirs)
        for sample_durs in tqdm(tasks, dataset_name, len(speaker_dirs), unit="speakers"):
            for sample_dur in sample_durs:
                logger.add_sample(duration=sample_dur)

    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)


def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing=False):
    for dataset_name in librispeech_datasets["train"]["other"]:
        # Inicializar el preprocesamiento
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Preprocesar todos los oradores 
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)


def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing=False):
    # Inicializar el preprocesamiento
    dataset_name = "VoxCeleb1"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Obtener el contenido del metaarchivo
    with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]

    # Seleccione la identificación y la nacionalidad, filtre los hablantes no anglófonos
    nationalities = {line[0]: line[3] for line in metadata}
    keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if
                        nationality.lower() in anglophone_nationalites]
    print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." %
          (len(keep_speaker_ids), len(nationalities)))

    # Obtenga los directorios de oradores solo para hablantes anglófonos
    speaker_dirs = dataset_root.joinpath("wav").glob("*")
    speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
                    speaker_dir.name in keep_speaker_ids]
    print("VoxCeleb1: found %d anglophone speakers on the disk, %d missing (this is normal)." %
          (len(speaker_dirs), len(keep_speaker_ids) - len(speaker_dirs)))

    # Preprocesar todos los oradores
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)


def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing=False):
   
    # Inicializar el preprocesamiento
    dataset_name = "VoxCeleb2"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

   # Obtener los directorios de oradores
    # Preprocesar todos los oradores
    speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)
