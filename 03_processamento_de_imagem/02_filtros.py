import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── FUNÇÃO: adicionar ruído sal e pimenta ────────────────────────────────────
def adicionar_ruido_sal_pimenta(img, intensidade=0.02):
    """
    Simula o ruído "sal e pimenta": pixels aleatórios viram branco (sal)
    ou preto (pimenta). Muito comum em sensores com falhas pontuais.

    intensidade: proporção de pixels afetados (0.02 = 2% da imagem)
    """
    img_ruidosa = img.copy()
    total_pixels = img.size // img.shape[2]  # total de pixels (sem contar canais)

    quantidade = int(total_pixels * intensidade)

    # Pixels brancos (sal)
    ys = np.random.randint(0, img.shape[0], quantidade)
    xs = np.random.randint(0, img.shape[1], quantidade)
    img_ruidosa[ys, xs] = [255, 255, 255]

    # Pixels pretos (pimenta)
    ys = np.random.randint(0, img.shape[0], quantidade)
    xs = np.random.randint(0, img.shape[1], quantidade)
    img_ruidosa[ys, xs] = [0, 0, 0]

    return img_ruidosa

# ── FUNÇÃO: filtro de média ───────────────────────────────────────────────────
def filtro_media(img, tamanho_kernel=5):
    """
    Substitui cada pixel pela MÉDIA aritmética dos seus vizinhos.
    O kernel é uma janela quadrada de 'tamanho_kernel x tamanho_kernel'.

    Vantagem:  simples e rápido.
    Desvantagem: borra bordas e detalhes junto com o ruído.
    tamanho_kernel deve ser ímpar: 3, 5, 7, ...
    """
    return cv2.blur(img, (tamanho_kernel, tamanho_kernel))


# ── FUNÇÃO: filtro gaussiano ─────────────────────────────────────────────────
def filtro_gaussiano(img, tamanho_kernel=5, sigma=0):
    """
    Suaviza aplicando um kernel com pesos em forma de sino (curva de Gauss).
    Pixels mais próximos do centro têm MAIS influência no resultado.

    Vantagem:  preserva bordas melhor que a média simples.
    sigma=0: OpenCV calcula o sigma automaticamente a partir do tamanho do kernel.
    tamanho_kernel deve ser ímpar: 3, 5, 7, ...
    """
    return cv2.GaussianBlur(img, (tamanho_kernel, tamanho_kernel), sigma)


# ── FUNÇÃO: filtro de mediana ────────────────────────────────────────────────
def filtro_mediana(img, tamanho_kernel=5):
    """
    Substitui cada pixel pelo valor MEDIANO dos vizinhos
    (valor do meio após ordenar todos os pixels da janela).
    Excelente para remover ruído sal e pimenta sem borrar bordas.

    Vantagem:  muito eficaz contra ruído impulsivo (sal e pimenta).
    Desvantagem: mais lento; tamanho_kernel deve ser ímpar.
    """
    return cv2.medianBlur(img, tamanho_kernel)


# ── FUNÇÃO: filtro bilateral ─────────────────────────────────────────────────
def filtro_bilateral(img, diametro=9, sigma_cor=75, sigma_espaco=75):
    """
    Filtro avançado que suaviza preservando bordas automaticamente.
    Considera DOIS fatores para calcular o peso de cada vizinho:
      - Distância espacial (sigma_espaco): quão longe está o vizinho
      - Diferença de cor   (sigma_cor):   quão diferente é a cor do vizinho

    Pixels com cor muito diferente (borda) recebem peso BAIXO → borda preservada.
    Pixels com cor similar recebem peso ALTO → suavizados.

    diametro:     tamanho da vizinhança analisada
    sigma_cor:    sensibilidade à diferença de cor (maior = mais suavização)
    sigma_espaco: sensibilidade à distância espacial (maior = vizinhança maior)
    """
    return cv2.bilateralFilter(img, diametro, sigma_cor, sigma_espaco)


# ── FUNÇÃO: comparar filtros em grade ────────────────────────────────────────
def comparar_filtros(img_original, titulo_ruido="Imagem com Ruído"):
    """
    Exibe numa grade 2x3 a imagem original e os 4 filtros aplicados.
    Útil para comparar visualmente o comportamento de cada abordagem.
    """
    img_rgb     = cv2.cvtColor(img_original,                  cv2.COLOR_BGR2RGB)
    media_rgb   = cv2.cvtColor(filtro_media(img_original),     cv2.COLOR_BGR2RGB)
    gauss_rgb   = cv2.cvtColor(filtro_gaussiano(img_original), cv2.COLOR_BGR2RGB)
    mediana_rgb = cv2.cvtColor(filtro_mediana(img_original),   cv2.COLOR_BGR2RGB)
    bilat_rgb   = cv2.cvtColor(filtro_bilateral(img_original), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3)

    axes[0, 0].imshow(img_rgb);     axes[0, 0].set_title("Original")
    axes[0, 2].imshow(media_rgb);   axes[0, 2].set_title("Filtro de Média (k=5)")
    axes[1, 0].imshow(gauss_rgb);   axes[1, 0].set_title("Filtro Gaussiano (k=5)")
    axes[1, 1].imshow(mediana_rgb); axes[1, 1].set_title("Filtro de Mediana (k=5)")
    axes[1, 2].imshow(bilat_rgb);   axes[1, 2].set_title("Filtro Bilateral (d=9)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ── FUNÇÃO: demonstrar efeito do tamanho do kernel ───────────────────────────
def demonstrar_tamanhos_kernel(img, filtro_fn, titulo_filtro="Gaussiano"):
    """
    Aplica o mesmo filtro com kernels de tamanhos crescentes (3, 7, 15, 31)
    para mostrar como o tamanho do kernel afeta o nível de suavização.

    filtro_fn:     função de filtro a ser demonstrada (ex.: filtro_gaussiano)
    titulo_filtro: nome do filtro para exibir no título do gráfico
    """
    tamanhos = [3, 7, 15, 31]

    fig, axes = plt.subplots(1, len(tamanhos) + 1, figsize=(20, 4))

    # Imagem original como referência
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, k in enumerate(tamanhos):
        resultado = filtro_fn(img, k)
        axes[i + 1].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"Kernel {k}x{k}")
        axes[i + 1].axis("off")

    plt.suptitle(f"Efeito do Tamanho do Kernel — Filtro {titulo_filtro}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CÓDIGO DE TESTE
# ═══════════════════════════════════════════════════════════════════════════════

# ── FUNÇÃO: cortar região de interesse ────────────────────────────────────
def cortar(img, x, y, largura, altura):
    """
    Recorta uma região retangular da imagem.
    x, y: canto superior esquerdo do recorte (coluna, linha)
    Nota: o acesso a arrays NumPy é [linha:linha, coluna:coluna] (y antes de x)
    """
    return img[y : y+altura, x : x+largura]


# Teste 1: primeira imagem - comparar todos os filtros
image = cv2.imread("img/chris-curry.png")
image = cortar(image, 200, 200, 300, 300)  # recorta para facilitar visualização dos filtros
comparar_filtros(image)

# Teste 2: segunda imagem - comparar todos os filtros
image = cv2.imread("img/alexandre-daoust.png")
image = cortar(image, 200, 250, 300, 350)  # recorta para  facilitar visualização dos filtros
comparar_filtros(image)

# Teste 3: segunda imagem com ruído salt n pepper - comparar todos os filtros
img_sp = adicionar_ruido_sal_pimenta(image, intensidade=0.01)
comparar_filtros(img_sp)

# Teste 3: efeito do tamanho do kernel no filtro gaussiano 
# Kernels maiores = mais suavização, mas mais perda de detalhes
demonstrar_tamanhos_kernel(image, filtro_gaussiano, "Gaussiano")

# Teste 4: efeito do tamanho do kernel no filtro de mediana
# Kernels muito grandes começam a distorcer bordas e formas
demonstrar_tamanhos_kernel(image, filtro_mediana, "Mediana")