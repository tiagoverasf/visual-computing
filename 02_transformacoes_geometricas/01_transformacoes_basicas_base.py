import io                           # Trabalhar com dados na memória
import cv2                          # Processamento e análise de imagens e vídeos
import numpy as np                  # Trabalhar com arrays/matrizes e cálculos numéricos
import urllib.request               # Baixar/acessar arquivos através de URLs
from PIL import Image               # Permite abrir e manipular imagens
import matplotlib.pyplot as plt     # Exibir imagens, gráficos e resultados

path = r"../assets/img/cats 2.jpg"
image = np.array(Image.open(path))


resized = cv2.resize(image,(100,100)) # Alterar o tamanho (largura e altura) de uma imagem.
height, width = image.shape[:2]
center = (width // 2, height // 2) # Ponto central usado para girar a imagem
rotation_matriz = cv2.getRotationMatrix2D(center,90,1) # Vai girar em torno do centro da imagem
rotated = cv2.warpAffine(image, rotation_matriz, (width, height)) # Transformação afim em uma imagem
cropped = image[250:600,100:450] # image[y_inicial:y_final, x_inicial:x_final]

plt.figure(figsize=(10,10)) # Define o tamanho de uma nova figura

plt.subplot(2,2,1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(2,2,2)
plt.imshow(resized)
plt.title('Resized Image')

plt.subplot(2,2,3)
plt.imshow(rotated)
plt.title('Rotated Image')

plt.subplot(2,2,4)
plt.imshow(cropped)
plt.title('Cropped Image')

plt.tight_layout()
plt.show()
