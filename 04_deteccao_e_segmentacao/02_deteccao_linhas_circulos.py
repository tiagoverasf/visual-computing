import cv2
import numpy as np
import matplotlib.pyplot as plt


def detectar_linhas(img_bgr, limiar=80, min_comprimento=50, max_gap=10):
    """
    Detecta linhas retas usando a Transformada de Hough Probabilística.

    limiar:          votos mínimos para aceitar uma linha
    min_comprimento: comprimento mínimo (px) de um segmento para ser aceito
    max_gap:         gap máximo (px) entre pontos para ainda serem da mesma linha
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(img_gray, 50, 200)
    linhas = cv2.HoughLinesP(
        bordas,
        rho=1,
        theta=np.pi / 180,
        threshold=limiar,
        minLineLength=min_comprimento,
        maxLineGap=max_gap,
    )
    return linhas  # array (N, 1, 4) no OpenCV 4.x / (N, 4) no 5.x, ou None


def detectar_circulos(img_bgr, dp=1.2, dist_min=30, param1=100, param2=30,
                      raio_min=10, raio_max=0):
    """
    Detecta círculos usando a Transformada de Hough para círculos.

    dp:       resolução do acumulador em relação à imagem (1 = mesma resolução)
    dist_min: distância mínima entre centros de círculos detectados
    param1:   limiar superior do Canny interno
    param2:   limiar de votos no acumulador (menor → mais círculos, mais falsos positivos)
    raio_min/raio_max: intervalo de raios aceitos (raio_max=0 → sem limite)
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 2)
    circulos = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=dist_min,
        param1=param1,
        param2=param2,
        minRadius=raio_min,
        maxRadius=raio_max,
    )
    return circulos  # array (1, N, 3) com [cx, cy, raio] ou None


def desenhar_linhas(img_bgr, linhas, fundo_preto=False, cor=(0, 255, 0), espessura=2):
    canvas = np.zeros_like(img_bgr) if fundo_preto else img_bgr.copy()
    cor_uso = (255, 255, 255) if fundo_preto else cor
    if linhas is not None:
        # OpenCV 4.x retorna (N, 1, 4) e o 5.x retorna (N, 4);
        # reshape(-1, 4) deixa sempre uma linha [x1, y1, x2, y2] por segmento
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            cv2.line(canvas, (x1, y1), (x2, y2), cor_uso, espessura)
    return canvas


def desenhar_circulos(img_bgr, circulos, fundo_preto=False, cor=(0, 0, 255), espessura=2):
    if fundo_preto:
        # Acumulador: cada círculo soma +1 nos pixels do seu contorno.
        # Regiões com mais sobreposições ficam mais claras → fácil limiarizar depois.
        canvas = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        if circulos is not None:
            for cx, cy, r in np.round(circulos[0]).astype(int):
                mascara = np.zeros_like(canvas)
                cv2.circle(mascara, (cx, cy), r, 1, espessura)
                canvas = cv2.add(canvas, mascara)   # satura em 255, não transborda
        return canvas  # canal único — use cmap="gray" no imshow

    canvas = img_bgr.copy()
    if circulos is not None:
        for cx, cy, r in np.round(circulos[0]).astype(int):
            cv2.circle(canvas, (cx, cy), r, cor, espessura)
            cv2.circle(canvas, (cx, cy), 3, cor, -1)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════

image = cv2.imread("assets/img/cereja.png")

image = cv2.medianBlur(image, 3)

# ── Detecção com parâmetros restritivos → só o que realmente existe ──────────
#
# Linhas: limiar alto + comprimento mínimo grande → rejeita segmentos curtos e fracos
# Círculos: param2 alto → acumulador exige mais votos para aceitar um círculo
#           dist_min grande → impede múltiplas detecções do mesmo círculo
#
linhas   = detectar_linhas(image,   limiar=10, min_comprimento=30, max_gap=50)
# testar o min 100 e o gap 5 e limiar 80

# Para círculos concêntricos:
#   dist_min pequeno  → permite detectar vários círculos com centro próximo
#   param2 alto       → só aceita candidatos com muitos votos no acumulador
#   raio_min/raio_max → limita o intervalo de tamanhos esperados na imagem
circulos = detectar_circulos(image, dp=1.2, dist_min=30, param1=20, param2=110, raio_min=50, raio_max=550)

img_linhas   = desenhar_linhas(image,   linhas,   fundo_preto=True)   # branco s/ fundo preto
img_circulos = desenhar_circulos(image, circulos, fundo_preto=True)   # branco s/ fundo preto

# Resultado combinado: linhas verdes + círculos vermelhos
img_resultado = desenhar_linhas(image,         linhas,   cor=(0, 255, 0))
img_resultado = desenhar_circulos(img_resultado, circulos, cor=(0, 0, 255))

n_linhas   = len(linhas)   if linhas   is not None else 0
n_circulos = len(circulos[0]) if circulos is not None else 0
print(f"Linhas detectadas: {n_linhas}  |  Círculos detectados: {n_circulos}")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(cv2.cvtColor(image,        cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(cv2.cvtColor(img_linhas,   cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Linhas ({n_linhas})")

axes[2].imshow(img_circulos, cmap="gray")
axes[2].set_title(f"Círculos ({n_circulos})")

axes[3].imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
axes[3].set_title("Resultado combinado")

for ax in axes:
    ax.axis("off")

plt.suptitle("Detecção de Linhas (Hough) e Círculos (Hough)", fontsize=14)
plt.tight_layout()
plt.show()