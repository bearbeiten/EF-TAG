import cv2

# gewünschte Webcam-Auflösung setzen
width, height = 1280, 720

# erster Video-Capture-Device wählen (ev. mehrere Webcams)
cam = cv2.VideoCapture(0)
cam.set(3, width)  # Höhe setzen
cam.set(4, height)  # Breite setzen

# Kamera-Loop
while cam.isOpened():
    # Einzelbild holen
    success, frame = cam.read()
    if not success:
        print("Webcam-Bild nicht verfügbar!")
        continue
    
    # Bild spiegeln
    frame = cv2.flip(frame, 1)

    # Bild anzeigen
    cv2.imshow("Webcam-Bild", frame)
    
    # Tastendruck «q» beendet Schleife
    if cv2.waitKey(20) == 113:
        break
    
# Webcam freigeben und Fenster schliessen
cam.release()
cv2.destroyAllWindows()

# Nochmals auf Tasteneingabe warten (Fix damit Fenster zugeht)
cv2.waitKey(20)