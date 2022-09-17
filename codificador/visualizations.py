from datetime import datetime
from time import perf_counter as timer

import numpy as np
import umap
import visdom

from codificador.data_objects.speaker_verification_dataset import SpeakerVerificationDataset


colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255


class Visualizations:
    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=False):
        # Datos de seguimiento
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        print("Updating the visualizations every %d steps." % update_every)

        # si visdom esta desabilitado TODO: usar un mejor paradigma para eso
        self.disabled = disabled
        if self.disabled:
            return

        # Establecer el nombre del entorno
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        if env_name is None:
            self.env_name = now
        else:
            self.env_name = "%s (%s)" % (env_name, now)

        # Conéctese a visdom y abra la ventana correspondiente en el navegador
        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to "
                            "start it.")
        # webbrowser.open("http://localhost:8097/env/" + self.env_name)

        # Create the windows
        self.loss_win = None
        self.eer_win = None
        # self.lr_win = None
        self.implementation_win = None
        self.projection_win = None
        self.implementation_string = ""

    def log_params(self):
        if self.disabled:
            return
        from codificador import params_data
        from codificador import params_model
        param_string = "<b>Model parameters</b>:<br>"
        for param_name in (p for p in dir(params_model) if not p.startswith("__")):
            value = getattr(params_model, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        param_string += "<b>Data parameters</b>:<br>"
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        self.vis.text(param_string, opts={"title": "Parameters"})

    def log_dataset(self, dataset: SpeakerVerificationDataset):
        if self.disabled:
            return
        dataset_string = ""
        dataset_string += "<b>Speakers</b>: %s\n" % len(dataset.speakers)
        dataset_string += "\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={"title": "Dataset"})

    def log_implementation(self, params):
        if self.disabled:
            return
        implementation_string = ""
        for param, value in params.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string,
            opts={"title": "Training implementation"}
        )

    def update(self, loss, eer, step):
        # Actualizar los datos de seguimiento
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self.losses.append(loss)
        self.eers.append(eer)
        print(".", end="")

        # Actualice los gráficos cada <update_every> pasos
        if step % self.update_every != 0:
            return
        time_string = "Step time:  mean: %5dms  std: %5dms" % \
                      (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\nStep %6d   Loss: %.4f   EER: %.4f   %s" %
              (step, np.mean(self.losses), np.mean(self.eers), time_string))
        if not self.disabled:
            self.loss_win = self.vis.line(
                [np.mean(self.losses)],
                [step],
                win=self.loss_win,
                update="append" if self.loss_win else None,
                opts=dict(
                    legend=["Avg. loss"],
                    xlabel="Step",
                    ylabel="Loss",
                    title="Loss",
                )
            )
            self.eer_win = self.vis.line(
                [np.mean(self.eers)],
                [step],
                win=self.eer_win,
                update="append" if self.eer_win else None,
                opts=dict(
                    legend=["Avg. EER"],
                    xlabel="Step",
                    ylabel="EER",
                    title="Equal error rate"
                )
            )
            if self.implementation_win is not None:
                self.vis.text(
                    self.implementation_string + ("<b>%s</b>" % time_string),
                    win=self.implementation_win,
                    opts={"title": "Training implementation"},
                )

        # Restablecer el seguimiento
        self.losses.clear()
        self.eers.clear()
        self.step_times.clear()

    def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None, max_speakers=10):
        import matplotlib.pyplot as plt

        max_speakers = min(max_speakers, len(colormap))
        embeds = embeds[:max_speakers * utterances_per_speaker]

        n_speakers = len(embeds) // utterances_per_speaker
        ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        colors = [colormap[i] for i in ground_truth]

        reducer = umap.UMAP()
        projected = reducer.fit_transform(embeds)
        plt.scatter(projected[:, 0], projected[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP projection (step %d)" % step)
        if not self.disabled:
            self.projection_win = self.vis.matplot(plt, win=self.projection_win)
        if out_fpath is not None:
            plt.savefig(out_fpath)
        plt.clf()

    def save(self):
        if not self.disabled:
            self.vis.save([self.env_name])
