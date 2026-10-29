import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── FUNÇÃO: limiarização simples (threshold global) ───────────────────────────
def binarizar_simples(img_bgr, limiar=127):
    """
    Converte a imagem para preto e branco usando um limiar fixo global.
    Cada pixel é comparado com o limiar:
      - pixel > limiar  → branco (255)
      - pixel <= limiar → preto  (0)

    Funciona bem quando a iluminação é uniforme em toda a imagem.
    limiar: valor de corte entre 0 e 255
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # thresh: imagem binarizada
    # _:      o limiar usado (útil quando calculado automaticamente, como no Otsu)
    _, thresh = cv2.threshold(img_gray, limiar, 255, cv2.THRESH_BINARY)
    return thresh


# ── FUNÇÃO: limiarização inversa ──────────────────────────────────────────────
def binarizar_invertida(img_bgr, limiar=127):
    """
    Igual à binarização simples, porém com lógica invertida:
      - pixel > limiar  → preto  (0)
      - pixel <= limiar → branco (255)

    Útil quando o objeto de interesse é mais claro que o fundo,
    mas queremos objetos brancos sobre fundo preto para análise.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, limiar, 255, cv2.THRESH_BINARY_INV)
    return thresh


# ── FUNÇÃO: limiarização de Otsu ─────────────────────────────────────────────
def binarizar_otsu(img_bgr):
    """
    Calcula automaticamente o melhor limiar global analisando o histograma.
    O método de Otsu encontra o limiar que minimiza a variância dentro de
    cada classe (pixels escuros vs. pixels claros) — ou seja, maximiza a
    separação entre as duas classes.

    Ideal quando não sabemos qual limiar usar, ou quando ele muda entre imagens.
    Funciona melhor em imagens com histograma bimodal (dois picos bem separados).
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # THRESH_OTSU faz o OpenCV calcular o limiar ideal automaticamente
    # O valor passado (0) é ignorado — Otsu sobrescreve
    limiar_otsu, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Limiar calculado pelo Otsu: {limiar_otsu:.0f}")
    return thresh


# ── FUNÇÃO: comparar métodos de binarização ───────────────────────────────────
def comparar_binarizacoes(img_bgr):
    """
    Plota a imagem original em escala de cinza e os três métodos lado a lado
    para comparação direta de resultados.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Escala de Cinza (original)")

    axes[1].imshow(binarizar_simples(img_bgr, limiar=127), cmap="gray")
    axes[1].set_title("Limiar Global (127)")

    axes[2].imshow(binarizar_invertida(img_bgr, limiar=127), cmap="gray")
    axes[2].set_title("Limiar Invertido (127)")

    axes[3].imshow(binarizar_otsu(img_bgr), cmap="gray")
    axes[3].set_title("Otsu (limiar automático)")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Comparação de Métodos de Binarização", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── FUNÇÃO: efeito do limiar na binarização simples ──────────────────────────
def efeito_do_limiar(img_bgr):
    """
    Aplica a binarização simples com limiares crescentes para mostrar
    como o limiar controla quais regiões viram branco ou preto.

    Limiar baixo → maioria dos pixels vira branco (imagem clara).
    Limiar alto  → maioria dos pixels vira preto (imagem escura).
    """
    limiares = [50, 100, 127, 180, 220]

    fig, axes = plt.subplots(1, len(limiares) + 1, figsize=(22, 4))

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cmap="gray")
    axes[0].set_title("Cinza (original)")
    axes[0].axis("off")

    for i, t in enumerate(limiares):
        _, thresh = cv2.threshold(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), t, 255, cv2.THRESH_BINARY
        )
        axes[i + 1].imshow(thresh, cmap="gray")
        axes[i + 1].set_title(f"Limiar = {t}")
        axes[i + 1].axis("off")

    plt.suptitle("Efeito do Limiar na Binarização Global", fontsize=13)
    plt.tight_layout()
    plt.show()

# ── MEDIANA + OTSU: comparação antes e depois do filtro ──────────────────────
def comparar_otsu_mediana(img_bgr, ksize=10):
    """
    Compara o Otsu aplicado diretamente na imagem cinza versus após
    um filtro de mediana ksize×ksize.

    O filtro de mediana remove ruído sal-e-pimenta e suaviza texturas finas,
    tornando o histograma mais bimodal — o que em geral melhora o Otsu.
    ksize deve ser ímpar; se par, incrementa 1 para satisfazer o OpenCV.
    """
    if ksize % 2 == 0:
        ksize += 1  # cv2.medianBlur exige tamanho ímpar

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu SEM pré-processamento
    limiar_antes, otsu_antes = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Filtro de mediana 10×10 (arredondado para 11 se necessário)
    img_mediana = cv2.medianBlur(img_gray, ksize)

    # Otsu APÓS mediana
    limiar_depois, otsu_depois = cv2.threshold(
        img_mediana, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Limiar Otsu SEM mediana : {limiar_antes:.0f}")
    print(f"Limiar Otsu COM mediana : {limiar_depois:.0f}")

    _, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Cinza (original)")

    axes[1].imshow(otsu_antes, cmap="gray")
    axes[1].set_title(f"Otsu SEM mediana\n(limiar = {limiar_antes:.0f})")

    axes[2].imshow(img_mediana, cmap="gray")
    axes[2].set_title(f"Após mediana {ksize}×{ksize}")

    axes[3].imshow(otsu_depois, cmap="gray")
    axes[3].set_title(f"Otsu COM mediana\n(limiar = {limiar_depois:.0f})")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"Otsu: antes vs depois do filtro de mediana {ksize}×{ksize}", fontsize=13
    )
    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CÓDIGO DE TESTE
# ═══════════════════════════════════════════════════════════════════════════════

image = cv2.imread("img/mae-mu.jpg")

# Visão geral de todos os métodos
comparar_binarizacoes(image)

# Explora como o limiar muda o resultado na abordagem global
efeito_do_limiar(image)

comparar_otsu_mediana(image, ksize=71)