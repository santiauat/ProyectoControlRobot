"""
Sistema Integrado PLC-YOLO para Organización de Tubos
Punto de entrada principal del sistema
*** Versión Modificada para DOS CÁMARAS (Superior y Lateral) ***
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import json
import time
from pathlib import Path

# <<< Asumiendo que tus archivos están en estas carpetas >>>
from core.plc_controller import PLCController
from core.vision_processor_prueba import VisionProcessor
# <<< CAMBIO: Importar desde logger_prueba >>>
from utils.logger_prueba import setup_logger, log_resultado_procesamiento, log_estado_plc


class SistemaPLCYOLO:
    """
    Aplicación principal que integra:
    - Interfaz gráfica (Tkinter)
    - 2 Cámaras / Videos (Superior y Lateral)
    - 2 Modelos YOLO
    - Comunicación PLC
    """
    
    # <<< CORRECCIÓN: __init__ con doble guion bajo >>>
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema PLC-YOLO (Dual Cam) - Control de Tubos")
        # <<< CAMBIO: Tamaño aumentado para dos videos >>>
        self.root.geometry("1600x900") 
        
        # Logger
        Path('logs').mkdir(exist_ok=True) 
        self.logger = setup_logger('SistemaPLC', archivo_log='logs/sistema.log')
        self.logger.info("="*70)
        self.logger.info("INICIANDO SISTEMA PLC-YOLO (DUAL CAM)")
        self.logger.info("="*70)
        
        # Cargar configuración
        Path('config').mkdir(exist_ok=True) 
        self.config = self._cargar_configuracion()
        
        # Componentes del sistema
        self.controlador_plc = None
        self.vision_processor = None # Se inicializará al 'Iniciar'
        
        # Estado del sistema
        self.modo_realtime_activo = False
        self.modo_simulacion = self.config.get('sistema', {}).get('modo_simulacion', True)
        
        # <<< CAMBIO: Dos capturas de video >>>
        self.video_cap_sup = None
        self.video_cap_lat = None
        self.frame_actual_sup = None
        self.frame_actual_lat = None
        
        # <<< CAMBIO: Dos rutas de modelo >>>
        self.modelo_path_sup = None
        self.modelo_path_lat = None
        
        # UI
        self._crear_interfaz() # <<< Esta función AHORA SÍ existe >>>
        self._actualizar_estado_ui()
        
        self.logger.info("✅ Sistema inicializado correctamente")
    
    def _cargar_configuracion(self):
        """Carga configuración desde JSON"""
        # <<< CAMBIO: Usa el nombre de config correcto >>>
        config_path = 'config/plc_config_prueba.json'
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"✅ Configuración cargada desde {config_path}")
            
            # Asegurarse de que las secciones existan
            if 'plc' not in config: config['plc'] = {}
            if 'vision' not in config: config['vision'] = {}
            if 'sistema' not in config: config['sistema'] = {}

            return config
        except FileNotFoundError:
            self.logger.error(f"❌ CRÍTICO: No se encontró {config_path}. El sistema no funcionará.")
            messagebox.showerror("Error", f"No se encontró el archivo de configuración: {config_path}")
            self.root.destroy()
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error cargando {config_path}: {e}")
            messagebox.showerror("Error", f"No se pudo cargar config: {e}")
            self.root.destroy()
            return {}
    
    # <<< FUNCIÓN _crear_interfaz AÑADIDA >>>
    def _crear_interfaz(self):
        """Crea la interfaz gráfica"""
        
        # ==================== PANEL SUPERIOR ====================
        panel_superior = ttk.Frame(self.root)
        panel_superior.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(panel_superior, text="Sistema PLC-YOLO (Dual Cam)", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Sistema detenido")
        ttk.Label(panel_superior, textvariable=self.status_var, 
                 font=('Arial', 10)).pack(side=tk.RIGHT)
        
        # ==================== PANEL IZQUIERDO (Controles) ====================
        panel_controles = ttk.LabelFrame(self.root, text="Controles", padding=10)
        panel_controles.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # --- Sección: PLC ---
        ttk.Label(panel_controles, text="PLC", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.btn_conectar_plc = ttk.Button(panel_controles, text="🔌 Conectar PLC", 
                                          command=self._conectar_plc)
        self.btn_conectar_plc.pack(fill=tk.X, pady=5)
        self.btn_desconectar_plc = ttk.Button(panel_controles, text="Desconectar PLC", 
                                             command=self._desconectar_plc, state=tk.DISABLED)
        self.btn_desconectar_plc.pack(fill=tk.X, pady=5)
        self.plc_status_var = tk.StringVar(value="Desconectado")
        ttk.Label(panel_controles, textvariable=self.plc_status_var, 
                 foreground='red').pack(anchor=tk.W, pady=5)
        ttk.Separator(panel_controles, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # --- Sección: Modelo YOLO ---
        ttk.Label(panel_controles, text="Modelos YOLO", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # <<< CAMBIO: Botones para DOS modelos >>>
        ttk.Button(panel_controles, text="📁 Cargar Modelo Superior (.pt)", 
                  command=self._cargar_modelo_sup).pack(fill=tk.X, pady=5)
        self.modelo_sup_status_var = tk.StringVar(value="Sin modelo Sup.")
        ttk.Label(panel_controles, textvariable=self.modelo_sup_status_var).pack(anchor=tk.W, pady=2)
        
        ttk.Button(panel_controles, text="📁 Cargar Modelo Lateral (.pt)", 
                  command=self._cargar_modelo_lat).pack(fill=tk.X, pady=5)
        self.modelo_lat_status_var = tk.StringVar(value="Sin modelo Lat.")
        ttk.Label(panel_controles, textvariable=self.modelo_lat_status_var).pack(anchor=tk.W, pady=2)
        
        ttk.Separator(panel_controles, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # --- Sección: Cámara ---
        ttk.Label(panel_controles, text="Fuentes de Video", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # <<< CAMBIO: Botones para DOS videos >>>
        ttk.Button(panel_controles, text="📁 Cargar Video Superior", 
                  command=self._cargar_video_sup).pack(fill=tk.X, pady=5)
        self.camara_sup_status_var = tk.StringVar(value="Sin video Sup.")
        ttk.Label(panel_controles, textvariable=self.camara_sup_status_var).pack(anchor=tk.W, pady=2)
        
        ttk.Button(panel_controles, text="📁 Cargar Video Lateral", 
                  command=self._cargar_video_lat).pack(fill=tk.X, pady=5)
        self.camara_lat_status_var = tk.StringVar(value="Sin video Lat.")
        ttk.Label(panel_controles, textvariable=self.camara_lat_status_var).pack(anchor=tk.W, pady=2)

        ttk.Separator(panel_controles, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # --- Sección: Sistema ---
        ttk.Label(panel_controles, text="Sistema", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.chk_simulacion_var = tk.BooleanVar(value=self.modo_simulacion)
        self.chk_simulacion = ttk.Checkbutton(panel_controles, text="Modo Simulación (sin PLC)", 
                                             variable=self.chk_simulacion_var,
                                             command=self._toggle_simulacion)
        self.chk_simulacion.pack(anchor=tk.W, pady=5)
        self.btn_iniciar = ttk.Button(panel_controles, text="▶️ INICIAR SISTEMA", 
                                     command=self._iniciar_sistema, state=tk.DISABLED)
        self.btn_iniciar.pack(fill=tk.X, pady=10)
        self.btn_detener = ttk.Button(panel_controles, text="⏹️ DETENER", 
                                     command=self._detener_sistema, state=tk.DISABLED)
        self.btn_detener.pack(fill=tk.X, pady=5)
        
        # ==================== PANEL CENTRAL (Videos) ====================
        panel_videos = ttk.Frame(self.root)
        panel_videos.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # <<< CAMBIO: Dos canvas de video >>>
        panel_video_sup = ttk.LabelFrame(panel_videos, text="Vista Superior (Y, Conteo, QC)", padding=5)
        panel_video_sup.pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=(0,5))
        self.canvas_video_sup = tk.Canvas(panel_video_sup, bg='black')
        self.canvas_video_sup.pack(fill=tk.BOTH, expand=True)

        panel_video_lat = ttk.LabelFrame(panel_videos, text="Vista Lateral (Z, Seguridad)", padding=5)
        panel_video_lat.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM, pady=(5,0))
        self.canvas_video_lat = tk.Canvas(panel_video_lat, bg='black')
        self.canvas_video_lat.pack(fill=tk.BOTH, expand=True)
        
        # ==================== PANEL DERECHO (Resultados) ====================
        panel_resultados = ttk.LabelFrame(self.root, text="Logs y Resultados", padding=10)
        panel_resultados.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.text_resultados = tk.Text(panel_resultados, width=50, height=30, 
                                       font=('Consolas', 9))
        self.text_resultados.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(panel_resultados, command=self.text_resultados.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_resultados.config(yscrollcommand=scrollbar.set)
    
    def _conectar_plc(self):
        """Conecta al PLC"""
        try:
            self.controlador_plc = PLCController(self.config, self.logger) 
            if self.controlador_plc.conectar():
                self.plc_status_var.set("✅ Conectado")
                self.plc_status_var.config(foreground='green')
                self.btn_conectar_plc.config(state=tk.DISABLED)
                self.btn_desconectar_plc.config(state=tk.NORMAL)
                self._actualizar_estado_ui()
                self.logger.info("✅ PLC conectado exitosamente")
            else:
                self.plc_status_var.set("❌ Error Conexión")
                self.plc_status_var.config(foreground='red')
                messagebox.showerror("Error PLC", "No se pudo conectar al PLC. Revisa IP/Puerto y conexión.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al instanciar PLC: {e}")
            self.logger.error(f"❌ Error instanciando PLC: {e}", exc_info=True)
    
    def _desconectar_plc(self):
        """Desconecta del PLC"""
        if self.controlador_plc:
            self.controlador_plc.desconectar()
        self.plc_status_var.set("Desconectado")
        self.plc_status_var.config(foreground='red')
        self.btn_conectar_plc.config(state=tk.NORMAL)
        self.btn_desconectar_plc.config(state=tk.DISABLED)
        self._actualizar_estado_ui()
        self.logger.info("🔌 PLC desconectado")

    def _cargar_modelo_sup(self):
        archivo = filedialog.askopenfilename(title="Seleccionar modelo SUPERIOR", filetypes=[("Modelos YOLO", "*.pt")])
        if archivo:
            self.modelo_path_sup = archivo
            self.modelo_sup_status_var.set(f"✅ {Path(archivo).name}")
            self._actualizar_estado_ui()
            self.logger.info(f"Ruta modelo Superior: {archivo}")

    def _cargar_modelo_lat(self):
        archivo = filedialog.askopenfilename(title="Seleccionar modelo LATERAL", filetypes=[("ModelOS YOLO", "*.pt")])
        if archivo:
            self.modelo_path_lat = archivo
            self.modelo_lat_status_var.set(f"✅ {Path(archivo).name}")
            self._actualizar_estado_ui()
            self.logger.info(f"Ruta modelo Lateral: {archivo}")

    def _cargar_video_sup(self):
        archivo = filedialog.askopenfilename(title="Seleccionar video SUPERIOR", filetypes=[("Archivos de video", "*.mp4 *.avi *.mkv")])
        if archivo:
            if self.video_cap_sup: self.video_cap_sup.release()
            self.video_cap_sup = cv2.VideoCapture(archivo)
            if self.video_cap_sup.isOpened():
                fps = self.video_cap_sup.get(cv2.CAP_PROP_FPS)
                self.camara_sup_status_var.set(f"✅ {Path(archivo).name} ({int(fps)} FPS)")
                self._actualizar_estado_ui()
                self.logger.info(f"Video Superior cargado: {archivo}")
                ret, frame = self.video_cap_sup.read()
                if ret: self._mostrar_frame(frame, self.canvas_video_sup)
                self.video_cap_sup.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar
            else:
                messagebox.showerror("Error", f"No se pudo abrir el video Superior: {archivo}")
                self.logger.error(f"❌ No se pudo abrir el video Superior: {archivo}")

    def _cargar_video_lat(self):
        archivo = filedialog.askopenfilename(title="Seleccionar video LATERAL", filetypes=[("Archivos de video", "*.mp4 *.avi *.mkv")])
        if archivo:
            if self.video_cap_lat: self.video_cap_lat.release()
            self.video_cap_lat = cv2.VideoCapture(archivo)
            if self.video_cap_lat.isOpened():
                fps = self.video_cap_lat.get(cv2.CAP_PROP_FPS)
                self.camara_lat_status_var.set(f"✅ {Path(archivo).name} ({int(fps)} FPS)")
                self._actualizar_estado_ui()
                self.logger.info(f"Video Lateral cargado: {archivo}")
                ret, frame = self.video_cap_lat.read()
                if ret: self._mostrar_frame(frame, self.canvas_video_lat)
                self.video_cap_lat.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar
            else:
                messagebox.showerror("Error", f"No se pudo abrir el video Lateral: {archivo}")
                self.logger.error(f"❌ No se pudo abrir el video Lateral: {archivo}")

    def _toggle_simulacion(self):
        """Alterna modo simulación"""
        self.modo_simulacion = self.chk_simulacion_var.get()
        self._actualizar_estado_ui()
        self.logger.info(f"Modo simulación: {self.modo_simulacion}")
    
    def _actualizar_estado_ui(self):
        """Actualiza botones según estado del sistema"""
        plc_ok = (self.controlador_plc and self.controlador_plc.is_connected) or self.modo_simulacion
        modelos_ok = self.modelo_path_sup is not None and self.modelo_path_lat is not None
        videos_ok = (self.video_cap_sup is not None and self.video_cap_sup.isOpened()) and \
                    (self.video_cap_lat is not None and self.video_cap_lat.isOpened())
        
        puede_iniciar = plc_ok and modelos_ok and videos_ok
        
        if puede_iniciar:
            self.btn_iniciar.config(state=tk.NORMAL)
        else:
            self.btn_iniciar.config(state=tk.DISABLED)
    
    def _iniciar_sistema(self):
        """Inicia el loop principal del sistema"""
        if self.modo_realtime_activo:
            return
            
        try:
            self.logger.info("Inicializando VisionProcessor...")
            self.status_var.set("Cargando modelos...")
            self.root.update()
            
            self.vision_processor = VisionProcessor(
                self.config, 
                self.logger,
                self.modelo_path_sup,
                self.modelo_path_lat
            )
            
            if not self.vision_processor.modelos_cargados:
                messagebox.showerror("Error", "No se pudieron cargar los modelos en VisionProcessor.")
                self.logger.error("❌ Fallo al cargar modelos en VisionProcessor.")
                self.status_var.set("Error de modelo")
                return
            
            self.logger.info("✅ VisionProcessor listo y modelos cargados.")
            
        except Exception as e:
            self.logger.error(f"❌ Error fatal al inicializar VisionProcessor: {e}", exc_info=True)
            messagebox.showerror("Error Crítico", f"No se pudo iniciar VisionProcessor: {e}")
            return
            
        self.modo_realtime_activo = True
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_detener.config(state=tk.NORMAL)
        self.status_var.set("🟢 Sistema ACTIVO - Monitoreando")
        self.logger.info("🚀 Sistema iniciado")
        
        self.btn_conectar_plc.config(state=tk.DISABLED)
        self.chk_simulacion.config(state=tk.DISABLED)
        
        self._calibrar_sistema()

    def _calibrar_sistema(self):
        """Ejecuta la calibración Y (Superior) antes de iniciar el loop"""
        if not self.video_cap_sup or not self.video_cap_sup.isOpened():
            self.logger.error("No se puede calibrar, video superior no cargado.")
            return

        ret, frame_sup_calib = self.video_cap_sup.read()
        if ret:
            self.logger.info("🔧 Iniciando calibración Y (Superior)...")
            self.status_var.set("Calibrando...")
            self.root.update()
            
            self.vision_processor.calibrar_y(frame_sup_calib)
            
            self.video_cap_sup.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.logger.info(f"Calibración finalizada. Centros Y: {self.vision_processor.X_CENTROS_IDEALES}")
            self.status_var.set("Calibrado. Iniciando loop...")
            
            self._loop_principal()
        else:
            self.logger.error("❌ Error: No se pudo leer el primer frame del video Superior para calibración Y.")
            messagebox.showerror("Error", "No se pudo leer el frame de calibración.")
            self._detener_sistema()

    
    def _detener_sistema(self):
        """Detiene el sistema"""
        self.modo_realtime_activo = False
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)
        self.status_var.set("Sistema detenido")
        self.logger.info("⏹️ Sistema detenido")
        
        if not (self.controlador_plc and self.controlador_plc.is_connected):
            self.btn_conectar_plc.config(state=tk.NORMAL)
        self.chk_simulacion.config(state=tk.NORMAL)
    
    def _loop_principal(self):
        """
        Loop principal del sistema - Implementa el handshake PLC
        *** VERSIÓN DUAL CAM ***
        """
        if not self.modo_realtime_activo:
            self.logger.info("Loop detenido por bandera 'modo_realtime_activo'")
            return
        
        delay_siguiente = self.config.get('sistema', {}).get('delay_lectura_plc_ms', 100)
        
        try:
            # 1. Capturar frames
            ret_sup, frame_sup = self.video_cap_sup.read()
            ret_lat, frame_lat = self.video_cap_lat.read()

            # Manejar fin de video (reiniciar)
            if not ret_sup:
                self.logger.info("Video Superior finalizado, reiniciando...")
                self.video_cap_sup.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_sup, frame_sup = self.video_cap_sup.read()
            if not ret_lat:
                self.logger.info("Video Lateral finalizado, reiniciando...")
                self.video_cap_lat.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_lat, frame_lat = self.video_cap_lat.read()

            if not ret_sup or not ret_lat:
                self.logger.error("Error en loop: No se pueden leer frames de los videos.")
                self._detener_sistema()
                messagebox.showerror("Error", "Se perdieron las fuentes de video.")
                return

            self.frame_actual_sup = frame_sup.copy()
            self.frame_actual_lat = frame_lat.copy()

            # Mostrar frames *originales*
            self._mostrar_frame(self.frame_actual_sup, self.canvas_video_sup)
            self._mostrar_frame(self.frame_actual_lat, self.canvas_video_lat)

            # 2. Consultar PLC (o simular)
            procesar = False
            if self.modo_simulacion:
                procesar = True
                delay_siguiente = self.config.get('sistema', {}).get('delay_simulacion_ms', 500)
            elif self.controlador_plc and self.controlador_plc.is_connected:
                procesar = self.controlador_plc.leer_solicitud_inspeccion()
                log_estado_plc(self.controlador_plc, self.logger, procesar)
            
            # 3. Procesar si hay solicitud
            if procesar:
                self.status_var.set("🔄 Procesando solicitud...")
                self.root.update()
                
                if not self.vision_processor:
                    self.logger.error("Error crítico: VisionProcessor no inicializado.")
                    self._detener_sistema()
                    return

                resultado = self.vision_processor.procesar_frames_dual(
                    self.frame_actual_sup,
                    self.frame_actual_lat
                )
                
                valido, advertencias = self.vision_processor.validar_resultado(resultado)
                if advertencias:
                    for adv in advertencias:
                        self.logger.warning(adv)
                
                log_resultado_procesamiento(resultado, self.logger)
                
                # Mostrar en UI (Frames anotados y logs)
                self._mostrar_frame(resultado['annotated_sup'], self.canvas_video_sup)
                self._mostrar_frame(resultado['annotated_lat'], self.canvas_video_lat)
                self._mostrar_resultado(resultado)
                
                # Enviar a PLC
                if not self.modo_simulacion and self.controlador_plc:
                    exito_plc = resultado['plc_success']
                    
                    exito_escritura = self.controlador_plc.escribir_resultados(
                        resultado['desviacion_y_mm'], # Desviación Z
                        resultado['filas'],           # Conteo de filas
                        exito_plc
                    )
                    
                    if not exito_escritura:
                        self.logger.error("❌ FALLO AL ESCRIBIR EN PLC")
                        self.plc_status_var.set("❌ Error Escritura")
                        self.plc_status_var.config(foreground='red')
                
                delay_siguiente = self.config.get('sistema', {}).get('delay_post_proceso_ms', 500)
            else:
                if not self.modo_simulacion:
                    self.status_var.set("🟢 Monitoreando PLC (esperando D28=99)")
            
            # 4. Siguiente iteración
            self.root.after(delay_siguiente, self._loop_principal)
            
        except Exception as e:
            self.logger.error(f"❌ Error fatal en loop principal: {e}", exc_info=True)
            self._detener_sistema()
            messagebox.showerror("Error de Ejecución", f"Error fatal en el sistema: {e}")
    
    def _mostrar_frame(self, frame, canvas):
        """Muestra frame en un canvas específico, redimensionando"""
        try:
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 640, 480 # Default
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            h, w, _ = frame_rgb.shape
            ratio = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            
            if new_w <= 0 or new_h <= 0: return
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            imagen = Image.fromarray(frame_resized)
            imagen_tk = ImageTk.PhotoImage(imagen)
            
            canvas.delete("all")
            x_offset = (canvas_width - new_w) // 2
            y_offset = (canvas_height - new_h) // 2
            
            canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=imagen_tk)
            
            if canvas == self.canvas_video_sup:
                self.canvas_video_sup.image = imagen_tk
            else:
                self.canvas_video_lat.image = imagen_tk
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error al mostrar frame: {e} (Canvas: {canvas})")

    
    def _mostrar_resultado(self, resultado):
        """Muestra resultado DUAL en panel de texto"""
        timestamp = time.strftime("%H:%M:%S")
        
        texto = f"\n{'='*45}\n"
        texto += f"[{timestamp}] RESULTADO PROCESAMIENTO DUAL\n"
        texto += f"{'='*45}\n"
        
        codigo_plc = resultado['codigo_respuesta_plc']
        if codigo_plc == 2: # PARADA
            texto += f"🛑 DIAGNÓSTICO: PARADA CRÍTICA (Lateral)\n"
        elif codigo_plc == 1: # FALLO QC
            texto += f"⚠️ DIAGNÓSTICO: FALLO QC / Corrección Y\n"
        else:
            texto += f"✅ DIAGNÓSTICO: OK\n"
            
        texto += f"--- SUPERIOR (QC, Y, Conteo) ---\n"
        texto += f"  • Filas Restantes: {resultado['filas']}\n"
        texto += f"  • Desviación Y: {resultado['desviacion_y_px']} px\n" # Mostrar en Píxeles
        
        texto += f"\n--- LATERAL (Z, Seguridad) ---\n"
        texto += f"  • Corrección Z: {resultado['correccion_z_cmm']} cMM\n"
        texto += f"  • Desviación (a PLC): {resultado['desviacion_y_mm']:.2f} mm\n" # Mostrar Z en mm
        texto += f"  • Log Z: {resultado['log_z']}\n"

        self.text_resultados.insert(tk.END, texto)
        self.text_resultados.see(tk.END)
    
    def cerrar(self):
        """Limpia recursos al cerrar"""
        self.logger.info("Iniciando cierre del sistema...")
        self._detener_sistema()
        
        if self.video_cap_sup:
            self.logger.info("Liberando video Superior...")
            self.video_cap_sup.release()
        if self.video_cap_lat:
            self.logger.info("Liberando video Lateral...")
            self.video_cap_lat.release()
        
        if self.controlador_plc:
            self.logger.info("Desconectando PLC...")
            self.controlador_plc.desconectar()
        
        self.logger.info("👋 Sistema cerrado")
        self.root.destroy()

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
# <<< CORRECCIÓN: __name__ con doble guion bajo >>>
if __name__ == "__main__":
    
    root = tk.Tk()
    app = SistemaPLCYOLO(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar)
    root.mainloop()