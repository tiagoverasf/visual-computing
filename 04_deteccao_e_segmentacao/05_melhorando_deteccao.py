import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── reutiliza a detecção por cor da aula 3.6 ─────────────────────────────────
def detectar_por_cor(img_bgr, h_min, h_max, s_min=50, s_max=255, v_min=50, v_max=255):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if h_min > h_max:
        m1 = cv2.inRange(img_hsv, (h_min, s_min, v_min), (179, s_max, v_max))
        m2 = cv2.inRange(img_hsv, (0,     s_min, v_min), (h_max, s_max, v_max))
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(img_hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))


def desenhar_caixas(img_bgr, mascara, area_min=500):
    resultado = img_bgr.copy()
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return resultado


# ── FUNÇÃO: aplicar operação morfológica ─────────────────────────────────────
def aplicar_morfologia(mascara, operacao, tamanho=5):
    """
    Operações morfológicas trabalham sobre a máscara binária,
    expandindo ou contraindo as regiões brancas (pixels detectados).

    erosao:     encolhe as regiões → elimina pontos isolados e ruído fino
    dilatacao:  expande as regiões → preenche buracos pequenos e conecta fragmentos
    abertura:   erosão seguida de dilatação → remove ruído sem mudar o tamanho geral
    fechamento: dilatação seguida de erosão → fecha buracos internos ao objeto

    tamanho: lado do kernel (maior = operação mais agressiva)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))

    ops = {
        "erosao":     lambda m: cv2.erode(m, kernel),
        "dilatacao":  lambda m: cv2.dilate(m, kernel),
        "abertura":   lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel),
        "fechamento": lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel),
    }
    return ops[operacao](mascara)


# ── FUNÇÃO: comparar máscaras antes e depois da morfologia ───────────────────
def comparar_morfologia(img_bgr, mascara, tamanho=10):
    """
    Plota a máscara bruta e cada operação morfológica lado a lado,
    mostrando o efeito de cada uma sobre os pixels detectados.
    """
    operacoes = ["erosao", "dilatacao", "abertura", "fechamento"]
    mascaras = {op: aplicar_morfologia(mascara, op, tamanho) for op in operacoes}

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(mascara, cmap="gray")
    axes[0].set_title("Máscara bruta\n(sem morfologia)")
    axes[0].axis("off")

    titulos = {
        "erosao":     "Erosão\n(remove ruído)",
        "dilatacao":  "Dilatação\n(preenche buracos)",
        "abertura":   "Abertura\n(erosão → dilatação)",
        "fechamento": "Fechamento\n(dilatação → erosão)",
    }
    for i, op in enumerate(operacoes):
        axes[i + 1].imshow(mascaras[op], cmap="gray")
        axes[i + 1].set_title(titulos[op])
        axes[i + 1].axis("off")

    plt.suptitle("Efeito das Operações Morfológicas na Máscara", fontsize=14)
    plt.tight_layout()
    plt.show()

    return mascaras


# ── FUNÇÃO: comparar detecção bruta vs morfologia melhorada ──────────────────
def comparar_deteccao(img_bgr, mascara, mascara_melhorada, label_melhorada="fechamento"):
    """
    Plota lado a lado:
      - Máscara bruta + caixas resultantes
      - Máscara após morfologia + caixas resultantes

    Mostra o impacto direto da morfologia na qualidade da detecção final.
    """
    bruta     = desenhar_caixas(img_bgr, mascara)
    melhorada = desenhar_caixas(img_bgr, mascara_melhorada)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0][0].imshow(mascara, cmap="gray")
    axes[0][0].set_title("Máscara bruta")
    axes[0][0].axis("off")

    axes[0][1].imshow(mascara_melhorada, cmap="gray")
    axes[0][1].set_title(f"Máscara após {label_melhorada}")
    axes[0][1].axis("off")

    axes[1][0].imshow(cv2.cvtColor(bruta, cv2.COLOR_BGR2RGB))
    axes[1][0].set_title("Detecção sem morfologia")
    axes[1][0].axis("off")

    axes[1][1].imshow(cv2.cvtColor(melhorada, cv2.COLOR_BGR2RGB))
    axes[1][1].set_title(f"Detecção com {label_melhorada}")
    axes[1][1].axis("off")

    plt.suptitle("Impacto da Morfologia na Detecção por Cor", fontsize=14)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CÓDIGO DE TESTE
# ═══════════════════════════════════════════════════════════════════════════════

image = cv2.imread("../assets/img/GuardaSol.png")

mascara = detectar_por_cor(image, h_min=10, h_max=20, s_min=80, s_max=255, v_min=80, v_max=255)

# Plot 1: comparar as 4 operações morfológicas sobre a máscara
mascaras = comparar_morfologia(image, mascara, tamanho=5)

# Plot 2: detecção bruta vs fechamento (geralmente o melhor para detecção de objetos)
comparar_deteccao(image, mascara, mascaras["fechamento"], label_melhorada="fechamento")