
# Clonación de voz en tiempo real
Este repositorio es una implementación de  guia que nos permite saber el funcionamiento (SV2TTS) con un codificador de voz que funciona en tiempo real. [Transfer Learning from Speaker Verification to
Síntesis de texto a voz de múltiples altavoces](https://arxiv.org/pdf/1806.04558.pdf) 

SV2TTS es un marco de aprendizaje profundo en tres etapas. En la primera etapa, se crea una representación digital de una voz a partir de unos pocos segundos de audio. En la segunda y tercera etapa, esta representación se usa como referencia para generar discurso dado un texto arbitrario


## Puede ser adaptado a este programa
 [CoquiTTS](https://github.com/coqui-ai/tts). Es un repositorio TTS bueno y actualizado dirigido a la comunidad ML. También puede hacer clonación de voz y más, como clonación entre idiomas o conversión de voz.



### 1. Requisitos de instalación
1. Se admiten tanto Windows como Linux. Se recomienda una GPU para el entrenamiento y la velocidad de inferencia, pero no es obligatoria.
2. Se recomienda Python 3.7. o superior debería funcionar.
3. Instale [ffmpeg](https://ffmpeg.org/download.html#get-packages). Esto es necesario para leer archivos de audio.
4. Instale [PyTorch](https://pytorch.org/get-started/locally/). Elija la última versión estable, su sistema operativo, su administrador de paquetes (pip por defecto) y finalmente elija cualquiera de las versiones propuestas de CUDA si tiene una GPU; de lo contrario, elija una CPU. Ejecute el comando dado.
5. Instale los requisitos restantes con `pip install -r requirements.txt`


### 2. Proceso de Instalacion detallada.
1.  conda activate VoiceCloning
2. CD C:/la extencion donde se guarda el proyecto
3. conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
5. pip install -r requirements.txt
6. python demo_toolbox.py


### 3. (Opcional) Configuración de prueba
Antes de descargar cualquier conjunto de datos, puede comenzar probando su configuración con:

`Python demo_cli.py`

Si pasan todas las pruebas, está listo para comenzar.
