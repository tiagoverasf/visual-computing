import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = r"../assets/img/cats 2.jpg"

# abre a imagem
image = np.array(Image.open(path))
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# transformada de fourier 2D
dft = np.fft.fft2(gray_image)
# move a frequência zero para o centro
dft_shift = np.fft.fftshift(dft)


# dimensões da imagem
rows, cols = gray_image.shape
# centro da imagem
crow, ccol = rows // 2, cols // 2


# filtragem passa-Baixa (desfoque)
low_mask = np.zeros((rows, cols), np.uint8) # cria uma máscara de zeros com as mesmas dimensões da imagem
low_mask[crow-20:crow+20, ccol-20:ccol+20] = 1 # coloca 1 no centro da matriz

# filtragem passa-alta (detecção de bordas)
high_mask = np.ones((rows, cols), np.uint8) # cria uma máscara de uns com as mesmas dimensões da imagem
high_mask[crow-20:crow+20, ccol-20:ccol+20] = 0 # colcoa 0 no centro da matriz

# aplicando o filtro
lowpass_dft_filtered = dft_shift * low_mask
highpass_dft_filtered = dft_shift * high_mask

# transformada inversa - volta do domínio da frequência para a imagem
lowpass_dft_inverse = np.fft.ifftshift(lowpass_dft_filtered)
#reverte o deslocamento aplicado anteriormente para retornar os componentes de frequência às suas posições originais
image_lowpass = np.fft.ifft2(lowpass_dft_inverse)
image_lowpass = np.uint8(np.abs(image_lowpass))

highpass_dft_inverse = np.fft.ifftshift(highpass_dft_filtered)
image_highpass = np.fft.ifft2(highpass_dft_inverse)
image_highpass = np.uint8(np.abs(image_highpass))


plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

plt.subplot(3, 1, 2)
plt.imshow(image_lowpass, cmap='gray')
plt.title('Low-pass')

plt.subplot(3, 1, 3)
plt.imshow(image_highpass, cmap='gray')
plt.title('High-pass')

plt.tight_layout()
plt.show()