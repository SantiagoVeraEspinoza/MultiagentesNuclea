from IPython import display
display.clear_output()

import supervision as sv
import ultralytics
import numpy as np
ultralytics.checks()

import os

detected_trucks = {}

# obtener el directorio actual
cwd = os.getcwd()

# foto que vamos a procesar
video_path1 = os.path.join(cwd, "..", "assets", "entrada.mp4")
video_path2 = os.path.join(cwd, "..", "assets", "centro_carga.mp4")

# ubicacion del modelo base de yolo
model_path = os.path.join(cwd, "..", "weights", "materials", "materials.pt")

# ubicacion de salida
output_path = os.path.join(cwd, "..", "output")

# ubicacion de video de salida ya con anotaciones
output_video_path1 = os.path.join(output_path, "entrada_out.mp4")
output_video_path2 = os.path.join(output_path, "centro_carga_out.mp4")


# cargar modelo
model = ultralytics.YOLO(model_path)

# anotador que nos permite aplicar las predicciones a la imagen para visualizarlas
annotator = sv.BoxAnnotator()

# byte tracker nos permite mantener track de los objetos que se van detectando en el video
byte_tracker = sv.ByteTrack()

# esta funcion se manda llamar por cada frame del video
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # inferencia y obtener predicciones
    results = model(frame)

    # como solo tenemos un frame, solo tomamos el primer resultado de la lista
    first_frame_results = results[0]

    # convertir las predicciones a un objeto de supervision que servira para visualizarlas
    detections = sv.Detections.from_ultralytics(first_frame_results)
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        # imprimimos la clase en texto y la confianza que el modelo tiene en su prediccion
        f"{model.model.names[class_id]} {confidence:0.2f} {track_id}"
        for _, _, confidence, class_id, track_id
        in detections
    ]
    
    for _, _, confidence, class_id, track_id in detections:
        detected_trucks[track_id] = True

    return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

sv.process_video(
    source_path=video_path1,
    target_path=output_video_path1,
    callback=callback,
)

entrance_trucks = len(detected_trucks)
detected_trucks = {}

sv.process_video(
    source_path=video_path2,
    target_path=output_video_path2,
    callback=callback,
)

loading_trucks = len(detected_trucks)

print(f"Entraron {entrance_trucks} y cargaron {loading_trucks}")