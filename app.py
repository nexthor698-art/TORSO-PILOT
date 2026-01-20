import cv2
import mediapipe as mp
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Configuración de MediaPipe Pose (IA de detección de cuerpo)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sensor')
def sensor():
    success, frame = cap.read()
    if not success:
        return jsonify({"torso": False})

    # Convertir a RGB para la IA
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    torso_valido = False

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Punto 11 y 12 son los hombros
        hombro_izq = lm[11]
        hombro_der = lm[12]
        
        # 1. Verificar visibilidad (que los hombros se vean bien)
        # 2. Verificar distancia (si están muy separados, está muy cerca de la cámara)
        dist_hombros = abs(hombro_izq.x - hombro_der.x)
        
        if hombro_izq.visibility > 0.7 and hombro_der.visibility > 0.7:
            if 0.15 < dist_hombros < 0.45: # Rango perfecto de torso
                torso_valido = True

    return jsonify({"torso": torso_valido})

if __name__ == '__main__':
    app.run(port=5000)
