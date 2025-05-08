import cv2
import mediapipe as mp

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Abrir vídeo local
video = cv2.VideoCapture('video.mp4')  # Substitua pelo caminho do seu vídeo

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o frame com MediaPipe Pose
    results = pose.process(frame_rgb)

    # Desenhar os pontos e conexões no corpo
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # Mostrar o vídeo com detecção de pose
    cv2.imshow("Pose Detection", frame)

    # Sair se apertar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar vídeo e janelas
video.release()
cv2.destroyAllWindows()
