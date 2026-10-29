# A Transformada de Fourier (FFT) serve para analisar uma imagem de uma maneira diferente.
# Em vez de olhar para a imagem pensando em pixels, ela permite olhar para a imagem pensando em frequências.

import cv2                          # Processamento e análise de imagens e vídeos
import numpy as np                  # Trabalhar com arrays/matrizes e cálculos numéricos
from PIL import Image               # Permite abrir e manipular imagens
import matplotlib.pyplot as plt     # Exibir imagens, gráficos e resultados

path = r"../assets/img/cats 2.jpg"
image = np.array(Image.open(path))

# converte para tons de cinza / cv2.cvtColor(): converte a imagem de um espaço de cores para outro.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Para calcular a transformada de Fourier 2D:
#     fft2() converte a imagem do domínio espacial para o domínio da frequência
#     fftshift() move os componentes de baixa frequência para o centro.
dft = np.fft.fft2(gray_image)
dft_shift = np.fft.fftshift(dft)
# Para visualizar o espectro de magnitude:
#     Como a transformada de Fourier gera números complexos, utiliza-se os valores absolutos (np.abs()) para uma visualização significativa.
#     A função np.log melhora a visibilidade, pois os valores brutos de magnitude variam bastante em escala.
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# Resultados
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)
plt.imshow(gray_image, cmap='gray')

plt.subplot(2,1,2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()