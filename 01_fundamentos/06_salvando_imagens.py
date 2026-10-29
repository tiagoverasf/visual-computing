import cv2
import os
import matplotlib.pyplot as plt
import zipfile

# ── FUNÇÃO: salvar imagem como JPEG ──────────────────────────────
def salvar_jpeg(img_bgr, caminho, qualidade=95):
    """
    Salva a imagem no formato JPEG com a qualidade especificada.
    JPEG é um formato com perda (lossy): quanto menor a qualidade,
    menor o arquivo, porém mais artefatos de compressão aparecem.

    qualidade: inteiro de 0 a 100 (95 = padrão, 0 = máxima compressão)
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, qualidade]
    cv2.imwrite(caminho, img_bgr, params)


# ── FUNÇÃO: salvar imagem como PNG ───────────────────────────────
def salvar_png(img_bgr, caminho, compressao=3):
    """
    Salva a imagem no formato PNG com o nível de compressão especificado.
    PNG é um formato sem perda (lossless): a qualidade visual nunca é reduzida,
    apenas o tempo de compressão e o tamanho do arquivo variam.

    compressao: inteiro de 0 a 9
    (0 = sem compressão/mais rápido, 9 = máxima compressão/mais lento)
    """
    params = [cv2.IMWRITE_PNG_COMPRESSION, compressao]
    cv2.imwrite(caminho, img_bgr, params)

# ── FUNÇÃO: tamanho do arquivo em kilobytes ──────────────────────
def tamanho_kb(caminho):
    """
    Retorna o tamanho do arquivo em KB.
    Útil para comparar o impacto de diferentes parâmetros de compressão.
    """
    return os.path.getsize(caminho) / 1024

# ── FUNÇÃO: plotar comparação de qualidades JPEG ─────────────────
def plotar_comparacao_jpeg(img_bgr, pasta, qualidades=(10, 50, 75, 95)):
    """
    Salva a imagem em múltiplas qualidades JPEG e plota todas lado a lado,
    exibindo o tamanho resultante de cada arquivo no título do subplot.
    Permite visualizar a relação entre qualidade visual e espaço em disco.
    """

    os.makedirs(pasta, exist_ok=True)
    n = len(qualidades)

    plt.figure(figsize=(4 * n, 5))
    plt.suptitle("Comparação de qualidade JPEG", fontsize=13)

    for i, q in enumerate(qualidades):
        caminho = os.path.join(pasta, f"jpeg_q{q:03d}.jpg")
        salvar_jpeg(img_bgr, caminho, qualidade=q)
        kb = tamanho_kb(caminho)

        ax = plt.subplot(1, n, i + 1)
        ax.imshow(cv2.cvtColor(cv2.imread(caminho),cv2.COLOR_BGR2RGB))
        ax.set_title(f"Qualidade {q}\n{kb:.1f} KB")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ── FUNÇÃO: plotar comparação de compressão PNG ─────────────────────
def plotar_comparacao_png(img_bgr, pasta, niveis=(0, 3, 6, 9)):
    """
    Salva a imagem em múltiplos níveis de compressão PNG e plota lado a lado.
    Como PNG é lossless, a imagem visual é idêntica em todos os casos —
    apenas o tamanho do arquivo e o tempo de escrita variam.
    """
    os.makedirs(pasta, exist_ok=True)
    n = len(niveis)

    plt.figure(figsize=(4 * n, 5))
    plt.suptitle("Comparação de compressão PNG (sem perda de qualidade)",fontsize=13)

    for i, nivel in enumerate(niveis):
        caminho = os.path.join(pasta, f"png_c{nivel}.png")
        salvar_png(img_bgr, caminho, compressao=nivel)
        kb = tamanho_kb(caminho)

        ax = plt.subplot(1, n, i + 1)
        ax.imshow(cv2.cvtColor(cv2.imread(caminho),cv2.COLOR_BGR2RGB))
        ax.set_title(f"Compressão {nivel}\n{kb:.1f} KB")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ── FUNÇÃO: resumo comparativo JPEG vs PNG ─────────────────────────
def plotar_resumo_tamanhos(pasta):
    """
    Lê todos os arquivos salvos na pasta e plota um gráfico de barras
    comparando o tamanho em KB de cada um.
    Ajuda a visualizar de forma consolidada o impacto de formato e compressão.
    """
    arquivos = sorted(
        [f for f in os.listdir(pasta) if f.endswith((".jpg", ".png"))],
        key=lambda f: os.path.getsize(os.path.join(pasta, f)),
    )

    nomes = [f.replace("_", " ").replace(".jpg", " (JPEG)").replace(".png", " (PNG)")for f in arquivos]
    tamanhos = [tamanho_kb(os.path.join(pasta, f))for f in arquivos]
    cores = ["steelblue" if f.endswith(".jpg") else "seagreen"for f in arquivos]

    plt.figure(figsize=(10, 5))
    barras = plt.barh(nomes, tamanhos, color=cores)
    plt.xlabel("Tamanho (KB)")
    plt.title("Tamanho dos arquivos salvos — JPEG vs PNG")

    for barra, kb in zip(barras, tamanhos):
        plt.text(barra.get_width() + 5,barra.get_y() + barra.get_height() / 2,
            f"{kb:.1f} KB", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

# ── EXECUÇÃO ────────────────────────────────────────────────────────

image = cv2.imread("../assets/img/alvan-nee.png")

if image is None:
    raise FileNotFoundError("Imagem não encontrada: img/alvan-nee.png")

PASTA_SAIDA = "img/saida"

# JANELA 1 - JPEG: mesma imagem salva em quatro qualidades diferentes
# Observe os artefatos nas bordas e regiões de alto contraste nas qualidades baixas
plotar_comparacao_jpeg(image, PASTA_SAIDA, qualidades=(10, 50, 75, 95))

# JANELA 2 - PNG: mesma imagem salva em quatro níveis de compressão
# A imagem é visualmente idêntica, só o tamanho do arquivo muda
plotar_comparacao_png( image, PASTA_SAIDA, niveis=(0, 3, 6, 9))

# JANELA 3 - Resumo: gráfico de barras comparando todos os arquivos gerados
plotar_resumo_tamanhos(PASTA_SAIDA)

# JANELA 4 - compactando a imagem em zip
with zipfile.ZipFile('img/saida/teste.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zip:
    zip.write('img/cereja.png')