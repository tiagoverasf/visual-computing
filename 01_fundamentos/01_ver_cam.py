import cv2
import time

cap = cv2.VideoCapture(0) # 0: câmera do notebook
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    # pegando a quantidade de frame por segundo (cada frame é uma imagem)
    fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
    prev_time = curr_time


    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera Aberta', frame)
    # fecha a tela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # liberar a câmera
cv2.destroyAllWindows()