import cv2
# Função VideoCapture
camera = cv2.VideoCapture("../assets/videos/kitten.mp4")

while True:
    _, frame = camera.read()
    cv2.imshow('Janela de Video', frame)

    # pressione 'q' para sair
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()