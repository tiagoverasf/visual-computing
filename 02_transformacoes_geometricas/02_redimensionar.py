from PIL import Image                                # permite abrir e manipular imagens:
import numpy as np


caminho_entrada = r"assets/img/cats 2.jpg"
imagem = Image.open(caminho_entrada).convert("RGB")  # converte a img em RGB - garante que a imagem tenha 3 canais
arr = np.array(imagem)

altura_orig, largura_orig = arr.shape[:2]            #trabalhar somente com 2 dimensões
print(f"Original: {largura_orig} x {altura_orig}px "f"(shape NumPy: {arr.shape})")

#2.Definir o novo tamanho (Largura,Altura) em pixels
nova_largura = 800
nova_altura = 600

# 3.Redimensionar em NumPy usando vizinho mais próximo
# para cada pixel novo, calcula qual pixel da imagem original ele "amostra"
linhas = (np.arange(nova_altura) * altura_orig /nova_altura).astype(int)
colunas = (np.arange(nova_largura) * largura_orig / nova_largura).astype(int)
arr_redimensionado = arr[linhas[:, None], colunas]

#4. Voltar para PIL só para salvar em disco
caminho_saida = r"assets/img/escalada.jpeg"
Image.fromarray(arr_redimensionado).save(caminho_saida)
print(f"Redimensionado: {nova_largura} x {nova_altura} px" f"(shape Numpy: {arr_redimensionado.shape})")
print(f"Imagem redimensionada e salva com sucesso!!")