import cv2
import matplotlib.pyplot as plt


# - FUNÇÃO: Ajustar brilho e contraste
def ajustar_brilho_contraste(img, alpha=1.0, beta=0):
    """
    Ajusta brilho e contraste da imagem
    alpha: fator de contraste (1.0 = original, >1 aumemnta, <1 diminui;
    beta: offset de brilho (0=original,>0 clareia, <0 escurece;
    fórmula aplicada pixel a pixel: novo = alpha * pixel *
    convertScaleAbs garante que os valores fiquem no range
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# - FUNÇÃO: redimensionar imagem
def redimensionar(img, largura=None, altura=None, escala=None):
    """
    Redimensionar a imagem de forma flexível
    - Se escala fornecida: aplica porcentagem (0.5 = 50% do tamanho)
    - Se largura e altura fornecidas: usa o tamanho absoluto
    - Se apenas largura ou altura fornecidas: mantém proporção automaticamente
    """
    h_orig, w_orig = img.shape[:2]  # shape = (altura, largura, canais)

    if escala is not None:
        nova_larg = int(w_orig * escala)
        nova_altura = int(h_orig * escala)
    elif largura is not None:
        nova_larg, nova_altura = largura, altura
    elif largura:
        nova_larg = largura
        nova_larg = int(h_orig * largura / w_orig)
    elif altura:
        nova_altura = altura
        nova_altura = int(w_orig * altura / h_orig)
    else:
        return img

    # INTER_AREA é ideial para reduzir, INTER_LINEAR: ideal para ampliar
    # imagem de 4 pixels e ir para 2, posso ter de 2 e ir para 1 e meio
    interpol = cv2.INTER_AREA if nova_larg < w_orig else cv2.INTER_LINEAR
    return (
        cv2.resize(img, (nova_larg, nova_altura), interpolation=interpol)
    )


# - FUNÇÃO: cortar região de interesse -
def cortar(img, x, y, largura, altura):
    """
    Recorta uma região retangular da imagem
    x,y: canto superior esquerdo do recorde (coluna, linha)
    Nota: o acesso ao arrays Numpy é [linha:linha, coluna:coluna]
    (y antes de x)
    """
    return img[y:y + largura, x:x + altura]

# - FUNÇÃO: rotacionar imagem
def rotacionar(img, angulo, escala=1.0):
    """
    Rotacionar a imagem em torno do seu centro
    ângulo: graus (positivo = sentido anti horário)
    escala: fator de zoom durante a rotação (1.0 = sem zoom)
    """

    h, w = img.shape[:2]
    centro = (w / 2, h / 2) # rotacionando w-larg, h-altura no Centro
    # da imagem em pixels
    M = cv2.getRotationMatrix2D(centro, angulo, escala)
    return cv2.warpAffine(img, M, (w, h))

image = cv2.imread("../assets/img/laranja.png")

# usa cada função definida acima e vamos plotar os resultados em subplots
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_brilho = ajustar_brilho_contraste(image, alpha=1.3, beta=30)
#aumenta a imagem OBS: TESTEM como 0.5 e 0
img_redium = redimensionar(image, escala=0.5)
img_corte = cortar(image, 100, 100, 800, 600) # x=100, y=100, largura=800
#vamos testear com 2100, 1200, 2500, 1600
img_rot = rotacionar(image,angulo=3.0)

plt.figure(figsize=(16, 10))

ax1 = plt.subplot(2,3,1)
ax1.imshow(image_rgb)
ax1.set_title(f"Original {image_rgb.shape}")
ax1.axis('off')

ax2 = plt.subplot(2,3,2)
ax2.imshow(cv2.cvtColor(img_brilho, cv2.COLOR_BGR2RGB))
ax2.set_title("Brilho / Contraste")
ax2.axis('off')

ax3 = plt.subplot(2,3,3)
ax3.imshow(cv2.cvtColor(img_corte, cv2.COLOR_BGR2RGB))
ax3.set_title("Corte")
ax3.axis('off')

ax4 = plt.subplot(2,3,4)
ax4.imshow(cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB))
ax4.set_title("Rotacionar")
ax4.axis('off')

ax5 = plt.subplot(2,3,5)
ax5.imshow(cv2.cvtColor(img_redium, cv2.COLOR_BGR2RGB))
ax5.set_title("Redimencionar")
ax5.axis('off')

plt.show()