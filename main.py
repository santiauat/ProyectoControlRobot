"""
Sistema Integrado PLC-YOLO para Organización de Tubos
Punto de entrada principal del sistema
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
from core.vision_processor import VisionProcessor
from utils.logger import setup_logger, log_resultado_procesamiento, log_estado_plc


class SistemaPLCYOLO:
    """
    Aplicación principal que integra:
    - Interfaz gráfica (Tkinter)
    - Cámara / Video
    - Modelo YOLO
    - Comunicación PLC
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema PLC-YOLO - Control de Tubos")
        self.root.geometry("1200x800")
        
        # Logger
        # <<< Asegúrate de que la carpeta 'logs' exista >>>
        Path('logs').mkdir(exist_ok=True) 
        self.logger = setup_logger('SistemaPLC', archivo_log='logs/sistema.log')
        self.logger.info("="*70)
        self.logger.info("INICIANDO SISTEMA PLC-YOLO")
        self.logger.info("="*70)
        
        # Cargar configuración
        # <<< Asegúrate de que la carpeta 'config' exista >>>
        Path('config').mkdir(exist_ok=True) 
        self.config = self._cargar_configuracion()
        
        # Componentes del sistema
        self.controlador_plc = None
        self.vision_processor = None
        self.modelo_yolo = None
        
        # Estado del sistema
        self.modo_realtime_activo = False
        self.modo_simulacion = self.config.get('sistema', {}).get('modo_simulacion', True) # <<< Lectura más segura >>>
        self.video_cap = None
        self.frame_actual = None
        
        # UI
        self._crear_interfaz()
        self._actualizar_estado_ui()
        
        self.logger.info("✅ Sistema inicializado correctamente")
    
    def _cargar_configuracion(self):
        """Carga configuración desde JSON"""
        config_path = 'config/plc_config.json'
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"✅ Configuración cargada desde {config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"⚠️ No se encontró {config_path}. Usando config vacía.")
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error cargando {config_path}: {e}")
            messagebox.showerror("Error", f"No se pudo cargar config: {e}")
            return {}
    
    def _crear_interfaz(self):
        """Crea la interfaz gráfica"""
        
        # ==================== PANEL SUPERIOR ====================
        panel_superior = ttk.Frame(self.root)
        panel_superior.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(panel_superior, text="Sistema PLC-YOLO", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Sistema detenido")
        ttk.Label(panel_superior, textvariable=self.status_var, 
                 font=('Arial', 10)).pack(side=tk.RIGHT)
        
        # ==================== PANEL IZQUIERDO (Controles) ====================
        panel_controles = ttk.LabelFrame(self.root, text="Controles", padding=10)
        panel_controles.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Sección: PLC
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
        
        # Sección: Modelo YOLO
        ttk.Label(panel_controles, text="Modelo YOLO", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        ttk.Button(panel_controles, text="📁 Cargar Modelo (.pt)", 
                  command=self._cargar_modelo).pack(fill=tk.X, pady=5)
        
        self.modelo_status_var = tk.StringVar(value="Sin modelo")
        ttk.Label(panel_controles, textvariable=self.modelo_status_var).pack(anchor=tk.W, pady=5)
        
        ttk.Separator(panel_controles, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección: Cámara
        ttk.Label(panel_controles, text="Fuente de Video", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # <<< CAMBIO: Botón y comando actualizados >>>
        ttk.Button(panel_controles, text="📁 Cargar Video", 
                  command=self._cargar_video).pack(fill=tk.X, pady=5)
        
        self.camara_status_var = tk.StringVar(value="Sin video") # <<< CAMBIO: Texto actualizado >>>
        ttk.Label(panel_controles, textvariable=self.camara_status_var).pack(anchor=tk.W, pady=5)
        
        ttk.Separator(panel_controles, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección: Sistema
        ttk.Label(panel_controles, text="Sistema", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        self.chk_simulacion_var = tk.BooleanVar(value=self.modo_simulacion) # <<< CAMBIO: Variable separada >>>
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
        
        # ==================== PANEL CENTRAL (Video) ====================
        panel_video = ttk.LabelFrame(self.root, text="Vista en Vivo", padding=10)
        panel_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_video = tk.Canvas(panel_video, width=640, height=480, bg='black')
        self.canvas_video.pack(fill=tk.BOTH, expand=True) # <<< CAMBIO: Permitir que se expanda >>>
        
        # ==================== PANEL DERECHO (Resultados) ====================
        panel_resultados = ttk.LabelFrame(self.root, text="Últimos Resultados", padding=10)
        panel_resultados.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.text_resultados = tk.Text(panel_resultados, width=40, height=30, 
                                       font=('Consolas', 9))
        self.text_resultados.pack(fill=tk.BOTH, expand=True, side=tk.LEFT) # <<< CAMBIO: Lado izquierdo >>>
        
        scrollbar = ttk.Scrollbar(panel_resultados, command=self.text_resultados.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_resultados.config(yscrollcommand=scrollbar.set)
    
    def _conectar_plc(self):
        """Conecta al PLC"""
        try:
            self.controlador_plc = PLCController(self.config) # <<< CAMBIO: Pasa la config >>>
            if self.controlador_plc.conectar():
                self.plc_status_var.set("✅ Conectado")
                self.plc_status_var.config(foreground='green') # <<< CAMBIO: Color >>>
                self.btn_conectar_plc.config(state=tk.DISABLED)
                self.btn_desconectar_plc.config(state=tk.NORMAL)
                self._actualizar_estado_ui()
                self.logger.info("✅ PLC conectado exitosamente")
            else:
                self.plc_status_var.set("❌ Error Conexión") # <<< CAMBIO: Más info >>>
                self.plc_status_var.config(foreground='red')
                messagebox.showerror("Error PLC", "No se pudo conectar al PLC. Revisa IP/Puerto y conexión.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al instanciar PLC: {e}")
            self.logger.error(f"❌ Error instanciando PLC: {e}")
    
    def _desconectar_plc(self):
        """Desconecta del PLC"""
        if self.controlador_plc:
            self.controlador_plc.desconectar()
            self.plc_status_var.set("Desconectado")
            self.plc_status_var.config(foreground='red') # <<< CAMBIO: Color >>>
            self.btn_conectar_plc.config(state=tk.NORMAL)
            self.btn_desconectar_plc.config(state=tk.DISABLED)
            self._actualizar_estado_ui()
            self.logger.info("🔌 PLC desconectado")
    
    def _cargar_modelo(self):
        """Carga modelo YOLO"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar modelo YOLO",
            filetypes=[("Modelos YOLO", "*.pt"), ("Todos", "*.*")]
        )
        
        if archivo:
            try:
                # <<< CAMBIO: Intenta importar YOLO aquí >>>
                from ultralytics import YOLO
                self.logger.info(f"Cargando modelo desde: {archivo}...")
                self.status_var.set("Cargando modelo...")
                self.root.update() # Forzar actualización de UI
                
                self.modelo_yolo = YOLO(archivo)
                
                # Crear processor
                self.vision_processor = VisionProcessor(self.config)
                self.vision_processor.modelo = self.modelo_yolo
                
                self.modelo_status_var.set(f"✅ {Path(archivo).name}")
                self._actualizar_estado_ui()
                self.logger.info(f"✅ Modelo cargado: {archivo}")
                self.status_var.set("Modelo cargado")
            except ImportError:
                messagebox.showerror("Error", "Librería 'ultralytics' no encontrada. Instálala con 'pip install ultralytics'")
                self.logger.error("❌ Error: Librería 'ultralytics' no encontrada.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
                self.logger.error(f"❌ Error cargando modelo: {e}")
                self.status_var.set("Error al cargar modelo")
    
    # <<< CAMBIO: Función reemplazada de _abrir_camara a _cargar_video >>>
    def _cargar_video(self):
        """Abre un archivo de video para procesar"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de video",
            filetypes=[("Archivos de video", "*.mp4 *.avi *.mkv *.mov"), ("Todos", "*.*")]
        )
        
        if archivo:
            try:
                # Si ya hay un video/cámara, liberarlo primero
                if self.video_cap:
                    self.video_cap.release()
                    
                self.video_cap = cv2.VideoCapture(archivo)
                
                if self.video_cap.isOpened():
                    # Obtener info del video
                    fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    frames = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    self.camara_status_var.set(f"✅ {Path(archivo).name} ({int(fps)} FPS)")
                    self._actualizar_estado_ui()
                    self.logger.info(f"✅ Video cargado: {archivo} ({int(frames)} frames @ {int(fps)} FPS)")
                    
                    # Mostrar primer frame
                    ret, frame = self.video_cap.read()
                    if ret:
                        self._mostrar_frame(frame)
                    # Resetear video al inicio
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                else:
                    messagebox.showerror("Error", f"No se pudo abrir el archivo de video: {archivo}")
                    self.logger.error(f"❌ No se pudo abrir el video: {archivo}")
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando video: {e}")
                self.logger.error(f"❌ Error cargando video: {e}")

    def _toggle_simulacion(self):
        """Alterna modo simulación"""
        self.modo_simulacion = self.chk_simulacion_var.get() # <<< CAMBIO: Lee desde la variable >>>
        self._actualizar_estado_ui()
        self.logger.info(f"Modo simulación: {self.modo_simulacion}")
    
    def _actualizar_estado_ui(self):
        """Actualiza botones según estado del sistema"""
        plc_ok = (self.controlador_plc and self.controlador_plc.is_connected) or self.modo_simulacion
        modelo_ok = self.modelo_yolo is not None
        camara_ok = self.video_cap is not None and self.video_cap.isOpened()
        
        puede_iniciar = plc_ok and modelo_ok and camara_ok
        
        if puede_iniciar:
            self.btn_iniciar.config(state=tk.NORMAL)
        else:
            self.btn_iniciar.config(state=tk.DISABLED)
    
    def _iniciar_sistema(self):
        """Inicia el loop principal del sistema"""
        if self.modo_realtime_activo:
            return # Ya está corriendo
            
        self.modo_realtime_activo = True
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_detener.config(state=tk.NORMAL)
        self.status_var.set("🟢 Sistema ACTIVO - Monitoreando")
        self.logger.info("🚀 Sistema iniciado")
        
        # Deshabilitar controles mientras corre
        self.btn_conectar_plc.config(state=tk.DISABLED)
        self.chk_simulacion.config(state=tk.DISABLED)
        
        # Iniciar loop
        self._loop_principal()
    
    def _detener_sistema(self):
        """Detiene el sistema"""
        self.modo_realtime_activo = False
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)
        self.status_var.set("Sistema detenido")
        self.logger.info("⏹️ Sistema detenido")
        
        # Rehabilitar controles
        if not (self.controlador_plc and self.controlador_plc.is_connected):
            self.btn_conectar_plc.config(state=tk.NORMAL)
        self.chk_simulacion.config(state=tk.NORMAL)
    
    def _loop_principal(self):
        """
        Loop principal del sistema - Implementa el handshake PLC
        """
        if not self.modo_realtime_activo:
            self.logger.info("Loop detenido por bandera 'modo_realtime_activo'")
            return
        
        delay_siguiente = 100  # ms (tiempo de espera entre lecturas de PLC)
        
        try:
            # 1. Capturar frame
            if self.video_cap and self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret and frame is not None:
                    self._mostrar_frame(frame)
                    self.frame_actual = frame.copy()
                else:
                    # <<< CAMBIO: Reiniciar video al terminar >>>
                    self.logger.info("Video finalizado, reiniciando...")
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar video
                    self.root.after(100, self._loop_principal) # Intentar leer de nuevo
                    return
            else:
                self.logger.error("Error en loop: VideoCap no está abierto.")
                self._detener_sistema()
                messagebox.showerror("Error", "Se perdió la fuente de video.")
                return

            # 2. Consultar PLC (o simular)
            procesar = False
            if self.modo_simulacion:
                procesar = True  # En simulación, procesar cada frame
            elif self.controlador_plc and self.controlador_plc.is_connected:
                procesar = self.controlador_plc.leer_solicitud_inspeccion()
                log_estado_plc(self.logger, self.controlador_plc, procesar) # <<< Log de estado >>>
            
            # 3. Procesar si hay solicitud
            if procesar:
                self.status_var.set("🔄 Procesando solicitud...")
                self.root.update()
                
                # Ejecutar YOLO
                if not self.vision_processor or not self.modelo_yolo:
                    self.logger.error("Error crítico: VisionProcessor o Modelo no están inicializados.")
                    self._detener_sistema()
                    return

                resultados_yolo = self.modelo_yolo.predict(self.frame_actual, verbose=False)
                
                # Procesar con VisionProcessor
                resultado = self.vision_processor.procesar_resultados(
                    resultados_yolo,
                    self.frame_actual.shape[1],  # ancho
                    self.frame_actual.shape[0]   # alto
                )
                
                # Validar
                valido, advertencias = self.vision_processor.validar_resultado(resultado)
                if advertencias:
                    for adv in advertencias:
                        self.logger.warning(adv)
                
                # Log
                log_resultado_procesamiento(resultado, self.logger)
                
                # Mostrar en UI
                self._mostrar_resultado(resultado)
                
                # Enviar a PLC
                if not self.modo_simulacion and self.controlador_plc:
                    exito_escritura = self.controlador_plc.escribir_resultados(
                        resultado['desviacion_mm'],
                        resultado['filas'],
                        resultado['success'] # 'success' ya es booleano
                    )
                    if not exito_escritura:
                        self.logger.error("❌ FALLO AL ESCRIBIR EN PLC")
                        self.plc_status_var.set("❌ Error Escritura")
                        self.plc_status_var.config(foreground='red')
                        # Podrías detener el sistema aquí si la escritura es crítica
                        # self._detener_sistema() 
                
                delay_siguiente = self.config.get('sistema', {}).get('delay_post_proceso_ms', 500) # Esperar más
            else:
                if not self.modo_simulacion:
                    self.status_var.set("🟢 Monitoreando PLC (esperando D28=99)")
            
            # 4. Siguiente iteración
            self.root.after(delay_siguiente, self._loop_principal)
            
        except Exception as e:
            self.logger.error(f"❌ Error fatal en loop principal: {e}", exc_info=True) # <<< exc_info=True >>>
            self._detener_sistema()
            messagebox.showerror("Error de Ejecución", f"Error fatal en el sistema: {e}")
    
    def _mostrar_frame(self, frame):
        """Muestra frame en canvas, redimensionando al tamaño del canvas"""
        try:
            # <<< CAMBIO: Redimensionar al tamaño del canvas dinámicamente >>>
            canvas_width = self.canvas_video.winfo_width()
            canvas_height = self.canvas_video.winfo_height()

            # Evitar error si el canvas es 0x0 al inicio
            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 640, 480 # Default
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (canvas_width, canvas_height))
            
            imagen = Image.fromarray(frame_resized)
            imagen_tk = ImageTk.PhotoImage(imagen)
            
            self.canvas_video.create_image(0, 0, anchor=tk.NW, image=imagen_tk)
            self.canvas_video.image = imagen_tk  # Mantener referencia
        except Exception as e:
            self.logger.warning(f"⚠️ Error al mostrar frame: {e}")

    
    def _mostrar_resultado(self, resultado):
        """Muestra resultado en panel de texto"""
        timestamp = time.strftime("%H:%M:%S")
        
        texto = f"\n{'='*40}\n"
        texto += f"[{timestamp}] RESULTADO\n"
        texto += f"{'='*40}\n"
        
        if resultado['success']:
            texto += f"✅ Estado: ÉXITO\n"
            texto += f"📊 Filas: {resultado['filas']}\n"
            texto += f"📏 Desviación: {resultado['desviacion_mm']:.2f} mm\n"
            
            if 'metadata' in resultado:
                meta = resultado['metadata']
                texto += f"\n📈 Metadata:\n"
                texto += f"  • Total detectado: {meta.get('total_detectado', 'N/A')}\n"
                texto += f"  • Válidas: {meta.get('detecciones_validas', 'N/A')}\n"
                texto += f"  • Conf. promedio: {meta.get('confianza_promedio', 0):.2%}\n"
        else:
            texto += f"❌ Estado: FALLO\n"
            texto += f"📋 Razón: {resultado.get('metadata', {}).get('razon_fallo', 'Desconocida')}\n"
        
        # Insertar al final y hacer scroll
        self.text_resultados.insert(tk.END, texto)
        self.text_resultados.see(tk.END)
    
    def cerrar(self):
        """Limpia recursos al cerrar"""
        self.logger.info("Iniciando cierre del sistema...")
        self._detener_sistema()
        
        if self.video_cap:
            self.logger.info("Liberando captura de video...")
            self.video_cap.release()
        
        if self.controlador_plc:
            self.logger.info("Desconectando PLC...")
            self.controlador_plc.desconectar()
        
        self.logger.info("👋 Sistema cerrado")
        self.root.destroy()


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    # <<< Asegúrate de que las carpetas core y utils estén accesibles >>>
    # (Esto asume que core/ y utils/ están al mismo nivel que main.py
    # o en el PYTHONPATH)
    
    root = tk.Tk()
    app = SistemaPLCYOLO(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar)
    root.mainloop()