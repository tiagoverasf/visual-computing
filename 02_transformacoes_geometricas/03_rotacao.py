from PIL import Image                                # permite abrir e manipular imagens
import numpy as np

# 1.Carregar a imagem e transformar em array NumPy
caminho_entrada = r"assets/img/cats 2.jpg"
imagem = Image.open(caminho_entrada)
arr = np.array(imagem)

# 2. Rotação de 90° no sentido HORÁRIO (basta usar k=-1)
arr_rotacionado = np.rot90(arr, k=-1)
print(f"Rotacionado: shape NumPy = {arr_rotacionado.shape}" f"(altura e largura trocam de lugar)")

# 3. Voltar para PIL só para salvar em disco
caminho_saida = r"assets/img/imagem_rotacionada.png"
Image.fromarray(arr_rotacionado).save(caminho_saida)

print("\033[30;47mImagem rotacionada e salva com sucesso!\033[m")
