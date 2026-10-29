import cv2
import matplotlib.pyplot as plt


def detectar_por_cor(img_bgr, h_min, h_max, s_min=50, s_max=255, v_min=50, v_max=255):
    """
    Detecta pixels de uma cor específica usando o espaço HSV.

    Por que HSV?
      No RGB, a mesma cor pode ter valores muito diferentes dependendo da
      iluminação. No HSV, H (matiz) define a cor independente do brilho —
      basta definir um intervalo de H para capturar uma cor inteira.

    H: matiz (0–179 no OpenCV)  → define a cor
    S: saturação (0–255)        → quão "viva" é a cor (baixo = cinza)
    V: valor/brilho (0–255)     → quão clara ou escura

    Vermelho é especial: fica em H≈0–10 E H≈170–179 (extremos do círculo).
    Se h_min > h_max, a função cria duas máscaras e as une automaticamente.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if h_min > h_max:
        # Vermelho e cores que cruzam H=0: dividir em dois intervalos
        m1 = cv2.inRange(img_hsv, (h_min, s_min, v_min), (179, s_max, v_max))
        m2 = cv2.inRange(img_hsv, (0,     s_min, v_min), (h_max, s_max, v_max))
        return cv2.bitwise_or(m1, m2)

    return cv2.inRange(img_hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))


def desenhar_caixas(img_bgr, mascara, area_min=500):
    """
    Encontra regiões conectadas na máscara e desenha uma caixa delimitadora
    (bounding box) em verde limão ao redor de cada detecção relevante.

    area_min: ignora regiões menores que esse número de pixels (filtra ruído).
    """
    resultado = img_bgr.copy()
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n = 0
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 10)
        n += 1

    print(f"Objetos detectados: {n}")
    return resultado


def plotar_deteccao(img_bgr, mascara):
    resultado = desenhar_caixas(img_bgr, mascara)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")

    axes[1].imshow(mascara, cmap="gray")
    axes[1].set_title("Pixels detectados (máscara HSV)")

    axes[2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Objetos detectados")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CÓDIGO DE TESTE
# ═══════════════════════════════════════════════════════════════════════════════

# Referência de matiz (H) para cores comuns no OpenCV (0–179):
#   Vermelho : 170–179 + 0–10  (usa h_min > h_max)
#   Laranja  : 10–20
#   Amarelo  : 20–35
#   Verde    : 35–85
#   Ciano    : 85–100
#   Azul     : 100–130
#   Roxo     : 130–160
#   Rosa/Mag : 160–170

"""
image = cv2.imread("img/mae-mu-vbAEHCrvXZ0-unsplash.jpg")
mascara = detectar_por_cor(image, h_min=170, h_max=10, s_min=80, s_max=255, v_min=80, v_max=255)
plotar_deteccao(image, mascara)
"""

image = cv2.imread("img/parmeet-singh-WXzfSzArC_c-unsplash.jpg")
mascara = detectar_por_cor(image, h_min=10, h_max=20, s_min=80, s_max=255, v_min=80, v_max=255)
plotar_deteccao(image, mascara)