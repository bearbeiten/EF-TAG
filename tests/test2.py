import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# GestureRecognizer-Objekt erstellen basieren auf heruntergeladenem Modell
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Liste von lokalen Bild-Dateien
files = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

# Schleife über alle Dateien
for file in files:
    # Bild aus Datei lesen
    image = mp.Image.create_from_file(file)

    # Bild durch Recognizer schicken
    recognition_result = recognizer.recognize(image)

    # Top-Geste der Erkennung auslesen
    top_gesture = recognition_result.gestures[0][0]
    
    # Ergebnis ausgeben
    print(file, top_gesture)