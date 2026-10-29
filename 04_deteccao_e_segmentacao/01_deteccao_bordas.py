import cv2
import matplotlib.pyplot as plt


def detectar_bordas_canny(img_bgr, limiar_baixo=50, limiar_alto=150):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, limiar_baixo, limiar_alto)


# ═══════════════════════════════════════════════════════════════════════════════

image = cv2.imread("../assets/img/cereja2.png")
#image = cv2.imread("img/espelho.png")

# filtros opcionais
image = cv2.medianBlur(image, 5)
image = cv2.GaussianBlur(image, (5, 5), 0)

"""
Limiar Superior (threshold2): Pixels com valor de gradiente maior que este valor são considerados bordas fortes e são aceitos imediatamente.
Limiar Inferior (threshold1): Pixels com valor de gradiente menor que este valor são considerados ruído e são descartados.
Zona Intermediária: Pixels com valor de gradiente entre o limiar inferior e superior são considerados bordas fracas. Eles só serão mantidos se estiverem conectados a um pixel de borda forte.
"""
bordas = detectar_bordas_canny(image, limiar_baixo=20, limiar_alto=50)
#testar com 200 no lim alto e 50 no baixo
# 80 e 20 também

# Overlay: dilata a máscara para ~5px de largura e pinta verde limão (veremos nas próx aulas)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mascara = cv2.dilate(bordas, kernel)
overlay = image.copy()
overlay[mascara == 255] = (0, 255, 0)  # BGR → verde limão

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(bordas, cmap="gray")
axes[1].set_title("Bordas (Canny)")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[2].set_title("Overlay verde limão")
axes[2].axis("off")

plt.suptitle("Detecção de Bordas — Canny", fontsize=14)
plt.tight_layout()
plt.show()