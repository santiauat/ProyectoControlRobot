"""
VisionProcessor_prueba - Módulo de procesamiento de visión artificial
*** VERSIÓN DUAL CAM (Superior + Lateral) ***
Integra la lógica de 'prueba_control.py'
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

class VisionProcessor:
    """
    Procesador de visión DUAL.
    Maneja la lógica de calibración, detección Superior (Y, QC, Conteo)
    y detección Lateral (Z, Seguridad).
    """
    
    # <<< CORRECCIÓN: __init__ con doble guion bajo >>>
    def __init__(self, 
                 config: Dict,
                 logger,
                 modelo_path_sup: str,
                 modelo_path_lat: str):
        """
        Inicializa el procesador de visión DUAL.
        
        Args:
            config: Diccionario de configuración (del JSON)
            logger: Instancia del logger
            modelo_path_sup: Ruta al modelo .pt Superior
            modelo_path_lat: Ruta al modelo .pt Lateral
        """
        self.config_vision = config.get('vision', {})
        self.logger = logger
        
        # --- Cargar constantes de 'prueba_control' desde config ---
        
        # General
        self.conf_sup = self.config_vision.get('confianza_sup', 0.45)
        self.conf_lat = self.config_vision.get('confianza_lat', 0.05)
        self.mm_per_pixel = self.config_vision.get('mm_per_pixel', 0.5) # Para conversión Y

        # Superior (Y, Conteo, QC)
        self.CLASES_FALLO_SUPERIOR = self.config_vision.get('CLASES_FALLO_SUPERIOR', ['error_apilado', 'error_alerta'])
        self.CLASE_POSICION = self.config_vision.get('CLASE_POSICION', 'posicion_columna')
        self.CLASE_VACIO = self.config_vision.get('CLASE_VACIO', 'posicion_vacia')
        self.TOTAL_POSICIONES = self.config_vision.get('TOTAL_POSICIONES', 8)
        self.TOLERANCIA_COLUMNA_PX = self.config_vision.get('TOLERANCIA_COLUMNA_PX', 30)
        self.CORRECCION_Y_FIJA_PX = self.config_vision.get('CORRECCION_Y_FIJA_PX', 50) # Límite
        
        # Lateral (Z, Seguridad)
        self.CLASES_ANOMALIA_LATERAL = self.config_vision.get('CLASES_ANOMALIA_LATERAL', ['error_caido'])
        self.CLASE_REFERENCIA = self.config_vision.get('CLASE_REFERENCIA', 'referencia_fija')
        self.CLASE_BORDE_ENV = self.config_vision.get('CLASE_BORDE_ENV', 'borde_envase')
        self.CLASE_MITAD_ENV = self.config_vision.get('CLASE_MITAD_ENV', 'mitad_envase')
        self.D_REAL_MM = self.config_vision.get('D_REAL_MM', 100.0)
        self.OFFSET_CERO_PX = self.config_vision.get('OFFSET_CERO_PX', 40)

        # Códigos de respuesta internos
        self.CODIGO_OK = 0
        self.CODIGO_FALLO_QC = 1
        self.CODIGO_PARADA = 2
        
        # Estado de calibración
        self.X_CENTROS_IDEALES = {}
        self.calibrado_y = False
        
        # Cargar modelos
        self.modelo_sup = None
        self.modelo_lat = None
        self.modelos_cargados = self._cargar_modelos(modelo_path_sup, modelo_path_lat)

    def _log(self, mensaje: str, nivel: str = 'info'):
        """Helper para loggear"""
        if self.logger:
            if nivel == 'info': self.logger.info(mensaje)
            elif nivel == 'warning': self.logger.warning(mensaje)
            elif nivel == 'error': self.logger.error(mensaje)
        else:
            print(mensaje) # Fallback a print

    def _cargar_modelos(self, path_sup, path_lat):
        """Carga los modelos YOLOv8 para ambas cámaras."""
        try:
            self._log(f"📦 Cargando modelo Superior desde {path_sup}...")
            self.modelo_sup = YOLO(path_sup)
            self._log(f"📦 Cargando modelo Lateral desde {path_lat}...")
            self.modelo_lat = YOLO(path_lat)
            self._log("✅ Modelos Superior y Lateral cargados.")
            return True
        except Exception as e:
            self._log(f"❌ ERROR al cargar modelos: {e}", 'error')
            return False

    def calibrar_y(self, frame_calibracion):
        """
        (Lógica de 'calcular_centros_ideales')
        Calcula la posición X ideal para cada una de las 8 columnas.
        """
        self._log("🔧 Iniciando calibración de centros ideales (eje Y)...")
        if not self.modelo_sup:
            self._log("❌ No se puede calibrar, modelo Superior no cargado.", 'error')
            return
            
        try:
            results = self.modelo_sup.predict(source=frame_calibracion, conf=0.1, verbose=False) # Confianza baja para calibrar
            
            centros_x_detectados = []
            
            for box in results[0].boxes:
                cls_name = self.modelo_sup.names.get(int(box.cls.item()))
                if cls_name == self.CLASE_POSICION or cls_name == self.CLASE_VACIO:
                    x_center = int((box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2)
                    centros_x_detectados.append(x_center)
                    
            if len(centros_x_detectados) < 2:
                self._log(f"⚠️ Calibración Y Fallida: Se necesitan al menos 2 columnas (detectadas: {len(centros_x_detectados)}).", 'warning')
                self.X_CENTROS_IDEALES = {}
                self.calibrado_y = False
                return

            centros_x_detectados.sort()
            deltas = [centros_x_detectados[i+1] - centros_x_detectados[i] 
                      for i in range(len(centros_x_detectados) - 1)]
            
            distancia_ideal_px = sum(deltas) / len(deltas)
            primer_centro_ideal = centros_x_detectados[0] 
            
            self.X_CENTROS_IDEALES = {}
            for i in range(self.TOTAL_POSICIONES):
                self.X_CENTROS_IDEALES[i + 1] = int(primer_centro_ideal + i * distancia_ideal_px)
                
            self.calibrado_y = True
            self._log(f"✅ Calibración Y Exitosa: Distancia promedio: {distancia_ideal_px:.2f} px")
            self._log(f"   Centros Ideales generados: {self.X_CENTROS_IDEALES}")

        except Exception as e:
            self._log(f"❌ Error durante calibración de centros Y: {e}", 'error')
            self.calibrado_y = False

    def _calcular_correccion_z(self, y_referencia, y_borde, y_mitad):
        """
        (Lógica de 'calcular_correccion_z')
        Calcula la corrección de altura (Eje Z) en centésimas de milímetro (cMM).
        """
        delta_p_escala = abs(y_borde - y_mitad) 
        
        if delta_p_escala == 0 or self.D_REAL_MM == 0:
            return 0, "No se pudo calcular la escala Z (Etiquetas 'borde' y 'mitad' colapsaron)."
            
        factor_escala_px_mm = delta_p_escala / self.D_REAL_MM
        delta_p_bruto = y_borde - y_referencia
        delta_p_error = delta_p_bruto - self.OFFSET_CERO_PX
        
        # cMM = (error_px / (px/mm)) * 10
        correccion_cmm = (delta_p_error / factor_escala_px_mm) * 10 
        
        return int(round(correccion_cmm)), None

    def _ejecutar_inferencia_lateral(self, frame_lat):
        """
        (Lógica de 'ejecutar_inferencia_lateral')
        Ejecuta inferencia en la cámara lateral (SEGURIDAD Y CORRECCIÓN Z).
        """
        results = self.modelo_lat.predict(source=frame_lat, conf=self.conf_lat, verbose=False) 
        annotated_lat = results[0].plot()
        
        response_code = self.CODIGO_OK
        correccion_z_cmm = 0 
        log_z = ""
        log_z_ref = ""
        
        y_coords = {self.CLASE_REFERENCIA: None, self.CLASE_BORDE_ENV: None, self.CLASE_MITAD_ENV: None}
        y_center_ref_fallback = frame_lat.shape[0] // 2
        
        # --- BÚSQUEDA DE DETECCIONES Y ANOMALÍAS ---
        for box in results[0].boxes:
            cls_name = self.modelo_lat.names.get(int(box.cls.item()))
            
            if cls_name in y_coords:
                y_center = int((box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2)
                # Tomar la detección con mayor confianza si hay múltiples
                if y_coords[cls_name] is None:
                     y_coords[cls_name] = y_center
                # (Se podría mejorar guardando la 'conf' y tomando el más alto)
                
            if cls_name in self.CLASES_ANOMALIA_LATERAL:
                self._log(f"🚨 Anomalía Lateral Crítica: {cls_name} detectada.", 'warning')
                response_code = self.CODIGO_PARADA
                log_z = f"PARADA CRÍTICA: {cls_name}"
                break 
            
        # --- CÁLCULO DE CORRECCIÓN Z ---
        if response_code != self.CODIGO_PARADA:
            if y_coords[self.CLASE_REFERENCIA] is None:
                 y_coords[self.CLASE_REFERENCIA] = y_center_ref_fallback
                 log_z_ref = "Usando centro imagen (Fallback Z)."

            if all(y_coords.values()):
                correccion_z_cmm, log_error = self._calcular_correccion_z(
                    y_coords[self.CLASE_REFERENCIA], 
                    y_coords[self.CLASE_BORDE_ENV], 
                    y_coords[self.CLASE_MITAD_ENV]
                )
                if log_error:
                    log_z = log_error
                    correccion_z_cmm = 0
                else:
                    log_z = f"Cálculo Z exitoso."
            else:
                # Loguear qué etiquetas faltaron
                faltantes = [k for k,v in y_coords.items() if v is None]
                log_z = f"Advertencia: Faltan etiquetas Z: {', '.join(faltantes)} (Z=0)."
                self._log(log_z, 'warning')
        
        log_final = log_z + (f" ({log_z_ref})" if log_z_ref else "")
        
        return response_code, annotated_lat, correccion_z_cmm, log_final

    def _ejecutar_inferencia_superior(self, frame_sup):
        """
        (Lógica de 'ejecutar_inferencia_superior')
        Ejecuta la detección superior: QC, Conteo y Corrección Y.
        """
        if not self.calibrado_y:
            self._log("Error: Inferencia superior llamada sin calibración Y.", 'error')
            # Retorna un fallo si no está calibrado
            annotated_sup = frame_sup.copy() # Devuelve frame original
            cv2.putText(annotated_sup, "ERROR: NO CALIBRADO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            return self.CODIGO_FALLO_QC, annotated_sup, 0, 0
            
        results = self.modelo_sup.predict(source=frame_sup, conf=self.conf_sup, verbose=False)
        annotated_sup = results[0].plot()
        
        has_qc_error = False
        detecciones_por_posicion = {} 
        
        for box in results[0].boxes:
            cls_name = self.modelo_sup.names.get(int(box.cls.item()))
            x_center = int((box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2)

            if cls_name in self.CLASES_FALLO_SUPERIOR:
                has_qc_error = True
                
            if cls_name == self.CLASE_VACIO:
                detecciones_por_posicion[x_center] = 'VACIO'
            elif cls_name == self.CLASE_POSICION:
                if x_center not in detecciones_por_posicion:
                    detecciones_por_posicion[x_center] = 'PRODUCTO'
        
        # --- CÁLCULO DE CONTEO Y COLUMNA DE TRABAJO ---
        conteo_filas_restantes = 0
        posicion_x_trabajo = None 
        
        posiciones_ordenadas = sorted(detecciones_por_posicion.keys())

        for x_pos in posiciones_ordenadas:
            estado = detecciones_por_posicion[x_pos]
            if estado == 'PRODUCTO':
                conteo_filas_restantes += 1
                if posicion_x_trabajo is None:
                    posicion_x_trabajo = x_pos
                    
        # --- CORRECCIÓN Y DINÁMICA ---
        correccion_y_pixels = 0
        
        if posicion_x_trabajo is not None:
            columna_actual_num = None
            min_dist = float('inf')
            
            for num_col, x_ideal in self.X_CENTROS_IDEALES.items():
                dist = abs(posicion_x_trabajo - x_ideal)
                if dist < min_dist:
                    min_dist = dist
                    columna_actual_num = num_col
            
            if columna_actual_num is not None:
                if min_dist > self.TOLERANCIA_COLUMNA_PX:
                    correccion_y_pixels = posicion_x_trabajo - self.X_CENTROS_IDEALES[columna_actual_num]
                    correccion_y_pixels = max(min(correccion_y_pixels, self.CORRECCION_Y_FIJA_PX), -self.CORRECCION_Y_FIJA_PX)
                    has_qc_error = True # Desviación = Fallo QC
            else:
                self._log("Error lógico: No se pudo mapear la posición de trabajo a una columna ideal.", 'error')
                has_qc_error = True
        
        # Determinar código de respuesta final
        response_code = self.CODIGO_OK
        if has_qc_error:
            response_code = self.CODIGO_FALLO_QC
        
        return response_code, annotated_sup, conteo_filas_restantes, correccion_y_pixels

    def procesar_frames_dual(self, frame_sup, frame_lat) -> Dict:
        """
        Función principal llamada por main.py.
        Ejecuta ambas inferencias y combina los resultados para el PLC.
        """
        
        # 1. Inferencia Lateral (Seguridad y Z)
        resp_lat_code, annotated_lat, correccion_z, log_z = \
            self._ejecutar_inferencia_lateral(frame_lat)
            
        # 2. Inferencia Superior (QC, Y, Conteo)
        # Solo se ejecuta si la lateral NO detectó una parada crítica
        if resp_lat_code != self.CODIGO_PARADA:
            resp_sup_code, annotated_sup, conteo, correccion_y_px = \
                self._ejecutar_inferencia_superior(frame_sup)
        else:
            # Si hay parada, no ejecutes la superior
            resp_sup_code = self.CODIGO_OK # No es un fallo de QC, es una parada
            annotated_sup = frame_sup.copy() # Devuelve el frame original
            cv2.putText(annotated_sup, "PARADA (LATERAL)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            conteo = 0
            correccion_y_px = 0

        # 3. Combinar resultados para el PLC
        
        # 'plc_success' es True SOLO si AMBAS cámaras están OK
        # (Un fallo de QC (código 1) SÍ se considera un 'success' para el PLC,
        # ya que el PLC debe recibir el código 1 para actuar (ej. rechazar))
        plc_success = (resp_lat_code != self.CODIGO_PARADA)
        
        # 'desviacion_y_mm' (para D29) viene de la corrección Y
        desviacion_y_mm = correccion_y_px * self.mm_per_pixel
        
        # 'filas' (para D14) viene del conteo
        filas = conteo
        
        # Determinar el código de respuesta general para el log
        # Prioridad: PARADA > FALLO_QC > OK
        if resp_lat_code == self.CODIGO_PARADA:
            codigo_respuesta_plc = self.CODIGO_PARADA
        elif resp_sup_code == self.CODIGO_FALLO_QC:
            codigo_respuesta_plc = self.CODIGO_FALLO_QC
        else:
            codigo_respuesta_plc = self.CODIGO_OK
            
        # *** LÓGICA DE MAPEO IMPORTANTE ***
        # Tu main.py envía 'desviacion_y_mm' y 'filas' al PLC.
        # ¿El PLC realmente espera la desviación Y en D29?
        # El script 'prueba_control' sugiere que la corrección Z (en cMM) 
        # podría ser el valor de desviación.
        
        # VAMOS A ASUMIR LA LÓGICA DE 'prueba_control':
        # D29 (desviacion_mm) = Corrección Z
        # D14 (filas) = Conteo Superior
        
        # <<< RE-MAPEAMOS LA SALIDA >>>
        desviacion_para_plc = correccion_z / 100.0 # Convertir cMM a MM
        
        return {
            'plc_success': plc_success,       # bool: Para D28 (88 u 77)
            'filas': filas,                 # int: Para D14
            'desviacion_y_mm': desviacion_para_plc, # <<< ¡ESTO AHORA ES LA CORRECCIÓN Z! >>>
            
            # --- Datos extra para logs y UI ---
            'correccion_z_cmm': correccion_z,
            'desviacion_y_px': correccion_y_px, # Desviación Y original en píxeles
            'codigo_respuesta_plc': codigo_respuesta_plc, 
            'annotated_sup': annotated_sup,   
            'annotated_lat': annotated_lat,
            'log_z': log_z,
        }

    def validar_resultado(self, resultado: Dict) -> Tuple[bool, List[str]]:
        """
        Valida el resultado DUAL antes de enviarlo al PLC.
        """
        advertencias = []
        
        if not resultado['plc_success']:
            # Los fallos ya están controlados (Parada)
            return True, [f"Fallo reportado (Código: {resultado['codigo_respuesta_plc']})"]
        
        # Validar desviación Z (la que se envía al PLC)
        desv_z = abs(resultado['desviacion_y_mm']) # Recordar que esto ahora es Z
        if desv_z > self.config_vision.get('max_correccion_z_mm_valida', 50.0): # 50mm
            advertencias.append(
                f"⚠️ Corrección Z muy grande: {resultado['desviacion_y_mm']:.2f}mm"
            )
        
        # Validar desviación Y (la que no se envía)
        desv_y = abs(resultado['desviacion_y_px'])
        if desv_y > (self.CORRECCION_Y_FIJA_PX + 1): # Si es mayor al límite
             advertencias.append(
                f"⚠️ Desviación Y fuera de rango: {resultado['desviacion_y_px']} px"
            )

        # Validar número de filas
        filas = resultado['filas']
        if filas < 0 or filas > self.TOTAL_POSICIONES:
            advertencias.append(f"⚠️ Número de filas fuera de rango: {filas} (esperado 0-{self.TOTAL_POSICIONES})")
        
        return len(advertencias) == 0, advertencias