import cv2
import matplotlib.pyplot as plt

# exibição utilizando OpenCV
imageColor = cv2.imread ("../assets/img/bernie-almanzar.png")
cv2.imshow("Mostrando a imagem", imageColor) # abre a janela com o titulo dado "Mostrando a imagem"
cv2.waitKey(0) # pausa o programa
# fecha todas as janelas abertas na tela pelo programa e libera a memória usada por elas
cv2.destroyAllWindows()

# exibição utilizando Matplotlib
img_rgb = cv2.cvtColor(imageColor, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)                 # exibe a imagem no plot
plt.title("Mostrando a imagem")     # define o titulo
plt.axis("off")                     # remove os eixos
plt.show()                          # renderiza a janela do matplotlib