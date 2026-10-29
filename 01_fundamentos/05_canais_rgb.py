import cv2
import matplotlib.pyplot as plt

# - FUNÇÃO: Separar canais RGB -
def separar_canais_rgb(img_brg):
    """
    Separar os 3 canais de uma RGB.
    Retorna (vermelho, verde, azul) como array 3D (escala de cinza)
    Nota: no OpenCV a ordem é BGR, então [:,:,0] = azul
    [:,:,1] = verde, [:,:,2] = vermelho matriz de 3 dimensões
    """

    #img[:,:,0] = primeiro canal = azul (blue)
    #img[:,:,1] = segundo canal = verde (green)
    #img[:,:,2] = terceiro canal = vermelho (red)

    azul = img_brg[:,:,0]
    verde = img_brg[:,:,1]
    vermelho = img_brg[:,:,2]

    return vermelho, verde, azul

# - FUNÇÃO: Visualização dos canais -
def visualizar_canais(img_bgr):
    """
    Plota a imagem original e seus 3 canais R, G, B lado a lado
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = separar_canais_rgb(img_bgr)
    fig, axes = plt.subplots(1,4, figsize=(16,4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (RGB)")

    axes[1].imshow(r, cmap="gray")
    axes[1].set_title("Canal (R) Vermelho")

    axes[2].imshow(g, cmap="gray")
    axes[2].set_title("Canal (G) Verde")

    axes[3].imshow(b, cmap="gray")
    axes[3].set_title("Canal (B) Blue")

    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.show()


#entendendo os vetores
imageColor = cv2.imread("../assets/img/bernie-almanzar.png", cv2.IMREAD_COLOR)
imageBlack = cv2.imread("../assets/img/bernie-almanzar.png", cv2.IMREAD_GRAYSCALE)

print("-> Conteúdo númerico da imagem imageBlack.png (escala cinza)")
print("Forma do vetor:", imageBlack.shape) #tamanho e formato da matriz
print("Primeiros 3x3 pixels (BGR)", imageColor[:3,:3,:]) # pega 3 pixes

# 3 dimensoes
#Exibiçõa de canais
# imageColor = cv2.imread("../assets/img/bernie-almanzar.png")
# visualizar_canais(imageColor)
# imageColor = cv2.imread("../assets/img/cereja.png")
# visualizar_canais(imageColor)
# imageColor = cv2.imread("../assets/img/connor-ward.png")
# visualizar_canais(imageColor)
# imageColor = cv2.imread("../assets/img/laranja.png")
# visualizar_canais(imageColor)
# imageColor = cv2.imread("../assets/img/GuardaSol.png")
# visualizar_canais(imageColor)

for nome in ["bernie-almanzar", "cereja", "connor-ward", "laranja", "GuardaSol"]:
    img = cv2.imread(f"../assets/img/{nome}.png")
    if img is None:
        print(f"Não achei {nome}.png")
        continue
    visualizar_canais(img)
