import cv2
import time
import psutil
import os
import torch
from inference import get_model

# MONITOREO NVIDIA 
try:
    import pynvml
    pynvml.nvmlInit()
    nvidia_disponible = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    nombre_gpu_raw = pynvml.nvmlDeviceGetName(handle)
    nombre_gpu = nombre_gpu_raw.decode('utf-8') if isinstance(nombre_gpu_raw, bytes) else nombre_gpu_raw
except Exception:
    nvidia_disponible = False
    nombre_gpu = "No detectada"

# Configuración inicial del modelo
model = get_model(model_id="weed-dectection/1", api_key="8Lv1kfdmOn7UDS3NIDTR")
cap = cv2.VideoCapture(2) # webcam logitech

# Obtener el proceso actual para medir su consumo
proceso = psutil.Process(os.getpid())
proceso.cpu_percent(interval=None)

historial_ram = []
historial_cpu = []
historial_hilos = []
historial_vram = []
historial_uso_gpu = []

prev_frame_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara")
            break

        new_frame_time = time.time()

        # Inferencia
        results = model.infer(frame)

        # Dibujar bounding boxes
        for pred in results[0].predictions:
            x, y, w, h = pred.x, pred.y, pred.width, pred.height
            x1 = int(x - (w/2))
            y1 = int(y - (h/2))
            x2 = int(x + (w/2))
            y2 = int(y + (h/2))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{pred.class_name} {pred.confidence:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calcular FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Deteccion de Maleza", frame)

        # RECOPILACIÓN DE MÉTRICAS CPU/RAM
        historial_ram.append(proceso.memory_info().rss / (1024 * 1024))
        historial_cpu.append(proceso.cpu_percent(interval=None))
        historial_hilos.append(proceso.num_threads())

        # RECOPILACIÓN DE MÉTRICAS GPU 
        if nvidia_disponible:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            historial_vram.append(mem_info.used / (1024 * 1024))
            historial_uso_gpu.append(util_info.gpu)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Finalizado")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Calcular estadísticas CPU/RAM/GPU
    if historial_ram and len(historial_cpu) > 0:
        promedio_ram = sum(historial_ram) / len(historial_ram)
        peak_ram = max(historial_ram)
        carga_bruta_cpu_promedio = sum(historial_cpu) / len(historial_cpu)
        nucleos_promedio = carga_bruta_cpu_promedio / 100.0
        nucleos_peak = max(historial_cpu) / 100.0
        peak_hilos = max(historial_hilos)
    else:
        promedio_ram = peak_ram = carga_bruta_cpu_promedio = 0
        nucleos_promedio = nucleos_peak = peak_hilos = 0

    if nvidia_disponible and len(historial_vram) > 0:
        promedio_vram = sum(historial_vram) / len(historial_vram)
        peak_vram = max(historial_vram)
        promedio_uso_gpu = sum(historial_uso_gpu) / len(historial_uso_gpu)
        peak_uso_gpu = max(historial_uso_gpu)
    else:
        promedio_vram = peak_vram = promedio_uso_gpu = peak_uso_gpu = 0

    nucleos_totales = psutil.cpu_count(logical=True)
    dispositivo_usado = "CUDA (NVIDIA GPU)" if torch.cuda.is_available() else "CPU"

    print("\n============================================================")
    print(" RESUMEN DE CONSUMO")
    print("============================================================")
    print(f" CPU Total:       {nucleos_totales} núcleos disponibles")
    print(f" Hilos Activos:     {peak_hilos} hilos (Peak en este nodo)")
    print("------------------------------------------------------------")
    print(f" NÚCLEOS USADOS:    {nucleos_promedio:.2f} núcleos (Promedio)")
    print(f" NÚCLEOS USADOS:    {nucleos_peak:.2f} núcleos (Peak)")
    print(f" Carga bruta CPU:   {carga_bruta_cpu_promedio:.2f} %")
    print("------------------------------------------------------------")
    print(f" RAM (Promedio):    {promedio_ram:.2f} MB")
    print(f" RAM (Peak):        {peak_ram:.2f} MB")
    print("------------------------------------------------------------")
    
    if nvidia_disponible:
        print(f" GPU Modelo:        {nombre_gpu}")
        print(f" VRAM Usada (Prom): {promedio_vram:.2f} MB")
        print(f" VRAM Usada (Peak): {peak_vram:.2f} MB")
        print(f" Carga GPU (Prom):  {promedio_uso_gpu:.2f} %")
        print(f" Carga GPU (Peak):  {peak_uso_gpu:.2f} %")
    else:
        print(" GPU NVIDIA:        No detectada o pynvml no instalado.")
        
    print("------------------------------------------------------------")
    print(f" Procesamiento IA:  Corriendo en {dispositivo_usado}")
    print("============================================================\n")
    
    if nvidia_disponible:
        pynvml.nvmlShutdown()