from codificador.params_model import *
from codificador.params_data import *
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn
import numpy as np
import torch


class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        
        # Definicion de Red 
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # Escalado de similitud de coseno (con valores de parámetros iniciales fijos)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # pérdida
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Recorte de degradado
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Calcula las incrustaciones de un lote de espectrogramas de pronunciación.
        
        :param expresiones: lote de bancos de filtros de escala Mel de la misma duración que un tensor de forma
        (tamaño_lote, n_fotogramas, n_canales)
        :param hidden_init: estado oculto inicial del LSTM como tensor de forma (num_layers,
        tamaño_lote, tamaño_oculto). Por defecto será un tensor de ceros si es Ninguno.
        :return: las incrustaciones como tensor de forma (batch_size, incrustar_size)
        """
        # Pase la entrada a través de las capas LSTM y recupere todas las salidas, el estado oculto final
        # y el estado final de la celda.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # solo se tomo estado oculto de la última capa.
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Calcula la matriz de similitud según la sección 2.1 de GE2E.

        :param embeds: las incrustaciones como tensor de forma (speakers_per_batch,
        expresiones_por_hablante, incrustación_tamaño)
        :return: la matriz de similitud como tensor de forma (speakers_per_batch,
        expresiones_por_hablante, hablantes_por_lote)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 por hablante). La clonación es necesaria para la diferenciación inversa
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusivos centroids (1 persona por enunciado)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # matriz de similitud. La similitud del coseno de vectores ya normados en 2 es simplemente el punto
        # producto de estos vectores (que es solo una multiplicación de elementos reducida por una suma).
        # Vectorizamos el cálculo por eficiencia.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Versión aún más vectorizada (más lenta quizás debido a la transposición)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
       :param embeds: las incrustaciones como tensor de forma (speakers_per_batch,
        expresiones_por_hablante, incrustación_tamaño)
        :return: la pérdida y el EER para este lote de incrustaciones.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # fragmento de  https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer
