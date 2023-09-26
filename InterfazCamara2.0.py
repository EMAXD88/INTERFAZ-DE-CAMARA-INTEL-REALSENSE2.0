import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import time
import os
import pyrealsense2 as rs
import numpy as np
from PIL import Image, ImageTk
import datetime

print("reset start")
ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
print("reset done")

# Función para iniciar la cámara RealSense
def start_camera():
    global pipeline, align, camera_running
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 5)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 5)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    camera_running = True

    # Bloquear la configuración de la cámara de profundidad para evitar modificaciones
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.enable_auto_exposure, False)
    depth_sensor.set_option(rs.option.enable_auto_white_balance, False)
    depth_sensor.set_option(rs.option.visual_preset, rs.option.visual_preset_custom)

# Función para tomar una foto
def take_photo():
    global camera_running, image_counter, previous_images
    if not camera_running:
        messagebox.showerror("ERROR", "LA CAMARA NO SE HA INICIADO CORRECTAMENTE!")
        return

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        messagebox.showerror("ERROR", "FALLO EN LA CAPTURA!")
        return

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Obtener la fecha actual en el formato YYYYMMDD
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    folder_name = os.path.join("CapturasIntelRealSense", current_date)
    os.makedirs(folder_name, exist_ok=True)

    # Crear subcarpetas para las imágenes RGB y de profundidad
    rgb_folder = os.path.join(folder_name, "RGB")
    depth_folder = os.path.join(folder_name, "Profundidad")
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    # Crear subcarpetas para las imágenes RGB y de profundidad en formato npy
    npy_rgb_folder = os.path.join(folder_name, "RGB_npy")
    npy_depth_folder = os.path.join(folder_name, "Profundidad_npy")
    os.makedirs(npy_rgb_folder, exist_ok=True)
    os.makedirs(npy_depth_folder, exist_ok=True)

    # Pedir al usuario que introduzca un nombre para la foto
    photo_name = simpledialog.askstring("Nombre de la foto", "Introduce un nombre para la foto:")

    if not photo_name:
        messagebox.showinfo("FOTO NO GUARDADA", "La foto no ha sido guardada debido a la falta de nombre.")
        return

    current_time = time.strftime("%H%M%S")
    rgb_image_path = os.path.join(rgb_folder, f"{photo_name}_{current_time}_rgb_{image_counter}.png")
    depth_image_path = os.path.join(depth_folder, f"{photo_name}_{current_time}_profundidad_{image_counter}.png")
    npy_rgb_image_path = os.path.join(npy_rgb_folder, f"{photo_name}_{current_time}_rgb_{image_counter}.npy")
    npy_depth_image_path = os.path.join(npy_depth_folder, f"{photo_name}_{current_time}_profundidad_{image_counter}.npy")

    cv2.imwrite(rgb_image_path, color_image)
    cv2.imwrite(depth_image_path, depth_image)

    np.save(npy_rgb_image_path, color_image)
    np.save(npy_depth_image_path, depth_image)

    # Agregar la imagen previa a la lista y mostrarla en una ventana flotante
    previous_images.append((photo_name, rgb_image_path))
    show_previous_images()

    messagebox.showinfo("FOTO TOMADA", "LA FOTO HA SIDO TOMADA Y GUARDADA!")

# Función para mostrar imágenes previas en una ventana flotante
def show_previous_images():
    global previous_images
    if not previous_images:
        return

    prev_image_window = tk.Toplevel(root)
    prev_image_window.title("Imágenes Previas")

    for name, path in previous_images:
        image = Image.open(path)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(prev_image_window, text=name)
        label.pack()
        label.img = photo
        label.config(image=photo)

# Función para cerrar la aplicación y detener la cámara
def close_app():
    global camera_running
    if camera_running:
        pipeline.stop()
    root.destroy()

# Función para actualizar las vistas de la cámara en tiempo real
def update_camera_frame():
    global camera_running
    if camera_running:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if color_frame and depth_frame:
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(color_image, (600, 400))))
            label_color.config(image=color_photo)
            label_color.image = color_photo

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image_rectified = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(depth_image_rectified*4, (600, 400))))
            label_depth.config(image=depth_photo)
            label_depth.image = depth_photo

# Creación de la ventana principal de la interfaz
root = tk.Tk()
root.title("INTERFAZ DE LA CAMARA")
root.attributes('-fullscreen', True)

# Variables globales para la cámara y su estado
pipeline = None
align = None
camera_running = False
image_counter = 0

# Creación del marco para las vistas de la cámara
frame_camera = tk.Frame(root)
frame_camera.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Etiqueta para la vista de la cámara en color (RGB)
label_color = tk.Label(frame_camera)
label_color.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Etiqueta para la vista de la cámara en profundidad
label_depth = tk.Label(frame_camera)
label_depth.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Creación del marco para los botones
frame_buttons = tk.Frame(root)
frame_buttons.pack(side=tk.TOP, padx=20, pady=20)

# Botón para iniciar la cámara
start_camera_btn = tk.Button(frame_buttons, text="INICIAR CAMARA", command=start_camera)
start_camera_btn.pack(side=tk.LEFT)

# Botón para tomar una foto
take_photo_btn = tk.Button(frame_buttons, text="TOMAR FOTO", command=take_photo)
take_photo_btn.pack(side=tk.LEFT, padx=10)

# Botón para salir de la aplicación
exit_btn = tk.Button(frame_buttons, text="SALIR", command=close_app)
exit_btn.pack(side=tk.LEFT, padx=10)

# Iniciamos la actualización de las vistas de la cámara con un retraso de 30 ms
root.after(60, update_camera_frame)
root.mainloop()
