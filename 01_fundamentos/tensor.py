from PIL import Image                       # permite abrir e manipular imagens:
import numpy as np

img = Image.open(r"../assets/img/cats 2.jpg")  # abrindo a imagem
arr = np.array(img)                         # transformando a imagem em array
print(arr.shape)                            # largura, altura, canais de cor