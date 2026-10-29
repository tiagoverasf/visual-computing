import cv2
import matplotlib.pyplot as plt

image_color = cv2.imread("../assets/img/bernie-almanzar.png")
img_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)


# plt.subplots cria de uma só vez uma figura (a janela ou imagem geral)
# e uma grade de eixos (os gráficos individuais)
# figsize() serve para definir a largura e a altura da imagem de um gráfico
fig, axes = plt.subplots(2, 3, figsize=(10,5))

# [linha,coluna]
#axes representa os eixos/áreas individuais de um gráfico
axes[0,0].imshow(img_rgb)               # coloca a img dentro da posição [0,0]
axes[0,0].set_title("fig 1 - LOCO")     # coloca um título

axes[0,1].imshow(img_rgb)
axes[0,1].set_title("fig 2 - LOC1")
axes[0,1].axis("off")                   # esconde os eixos

image_color = cv2.imread("../assets/img/connor-ward.png")
img_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# [linhas, colunas]
axes[1,1].imshow(img_rgb)               # coloca a img dentro da posição [1,1]
axes[1,1].set_title("fig 3 - L1C1")     # coloca um título

axes[1,2].imshow(img_rgb)
axes[1,2].set_title("fig 4 - L1C2")
axes[1,2].axis("off")

plt.figure(figsize=(10,5))              # cria uma nova janela ou um espaço em branco (chamado de figura)
                                        # e define o tamanho da imagem do gráfico

# 2 linha, 1 coluna
ax1 = plt.subplot(2,1,1)                # plt.subplot(linhas, colunas, posição)
ax1.imshow(img_rgb)
ax1.set_title(f'Original {img_rgb.shape}') # shape retornar as dimensões de uma imagem
ax1.axis("off")

ax1 = plt.subplot(2,1,2)                # plt.subplot(linhas, colunas, posição)
ax1.imshow(img_rgb)
ax1.set_title(f'Original {img_rgb.shape}') # shape retornar as dimensões de uma imagem
ax1.axis("off")

plt.tight_layout()                      # evita sobreposições
plt.show()                              # exibir na tela os gráficos