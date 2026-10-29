import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── FUNÇÃO: segmentação por cor via K-Means ───────────────────────────────────
def segmentar_kmeans(img_bgr, K, usar_mediana=True):
    """
    Segmenta a imagem agrupando os pixels em K clusters de cor.

    O K-Means funciona assim:
      1. Escolhe K centros aleatórios no espaço de cor (R, G, B)
      2. Cada pixel é associado ao centro mais próximo
      3. Os centros são recalculados como média dos pixels do cluster
      4. Repete até convergir

    Resultado: cada pixel recebe a cor média do seu cluster →
    a imagem fica com exatamente K cores distintas.

    K:            número de grupos de cor (2 = mínimo, 8–10 = muito detalhado)
    usar_mediana: suaviza a imagem antes do K-Means, reduz ruído e detalhes pequenos

    """
    img_proc = cv2.medianBlur(img_bgr, 5) if usar_mediana else img_bgr.copy()

    # K-Means exige array (N_pixels × 3) em float32
    pixels = img_proc.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    seg_bgr = centers[labels.flatten()].reshape(img_proc.shape)

    # Máscara binária de cada cluster (255 onde o pixel pertence ao cluster i)
    H, W = img_proc.shape[:2]
    label_map = labels.reshape(H, W)
    mascaras = [(label_map == i).astype(np.uint8) * 255 for i in range(K)]

    return seg_bgr, mascaras, centers


# ── FUNÇÃO: comparar diferentes valores de K lado a lado ─────────────────────
def comparar_k(img_bgr, valores_k=(2, 3, 5, 8)):
    """
    Plota a imagem original e a segmentação para cada valor de K,
    facilitando a escolha do número de clusters adequado para a imagem.

    K pequeno  → poucas cores, regiões grandes e simples
    K grande   → muitas cores, mais detalhes preservados
    """
    fig, axes = plt.subplots(1, len(valores_k) + 1, figsize=(5 * (len(valores_k) + 1), 5))

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, k in enumerate(valores_k):
        segmentada, _, centers = segmentar_kmeans(img_bgr, K=k)
        axes[i + 1].imshow(cv2.cvtColor(segmentada, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"K = {k}  ({k} cores)")
        axes[i + 1].axis("off")

    plt.suptitle("Comparação de K — Segmentação por Cor (K-Means)", fontsize=14)
    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CÓDIGO DE TESTE
# ═══════════════════════════════════════════════════════════════════════════════

image = cv2.imread("../assets/img/GuardaSol.png")
#image = cv2.imread("img/Laranja2.png")
#image = cv2.imread("img/carrinhos.png")

# 1. Comparar K=2,3,5,8 lado a lado
comparar_k(image, valores_k=(2, 3, 5, 8))