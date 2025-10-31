"""
VisionProcessor - Módulo de procesamiento de visión artificial
Integra YOLO con el sistema de control PLC
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO


class VisionProcessor:
    """
    Procesador de visión artificial para el sistema PLC-YOLO.
    
    Responsabilidades:
    - Procesar resultados de YOLO
    - Calcular métricas (desviación, número de filas)
    - Filtrar detecciones por confianza
    - Aplicar calibración espacial (mm/pixel)
    """
    
    def __init__(self, 
                 config: Dict,
                 modelo_path: str = None):
        """
        Inicializa el procesador de visión.
        
        Args:
            config: Diccionario de configuración (del JSON)
            modelo_path: Ruta al archivo .pt del modelo YOLO
        """
        self.config = config['vision']
        self.mm_per_pixel = self.config['mm_per_pixel']
        self.confianza_minima = self.config['confianza_minima']
        self.usar_centro_imagen = self.config['usar_centro_imagen']
        self.referencia_x_custom = self.config['referencia_x_custom']
        
        self.modelo = None
        if modelo_path:
            self.cargar_modelo(modelo_path)
    
    def cargar_modelo(self, modelo_path: str) -> bool:
        """
        Carga el modelo YOLO desde archivo.
        
        Args:
            modelo_path: Ruta al archivo .pt
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            print(f"📦 Cargando modelo YOLO desde {modelo_path}...")
            self.modelo = YOLO(modelo_path)
            print("✅ Modelo YOLO cargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def procesar_resultados(self, 
                           yolo_results,
                           ancho_imagen: int,
                           alto_imagen: int) -> Dict:
        """
        Procesa resultados de YOLO y calcula métricas para el PLC.
        
        Pipeline de procesamiento:
        1. Validar que hay detecciones
        2. Filtrar por confianza mínima
        3. Calcular desviación en mm
        4. Contar filas
        5. Generar metadata para debugging
        
        Args:
            yolo_results: Resultados crudos de model.predict()
            ancho_imagen: Ancho de la imagen en píxeles
            alto_imagen: Alto de la imagen en píxeles
            
        Returns:
            Dict con estructura:
            {
                'success': bool,
                'filas': int,
                'desviacion_mm': float,
                'metadata': dict
            }
        """
        result = yolo_results[0]
        
        # Validación: sin detecciones
        if result.boxes is None or len(result.boxes) == 0:
            return self._generar_respuesta_fallo(
                "No se detectaron objetos en la imagen"
            )
        
        # Filtrar detecciones válidas
        detecciones_validas = self._filtrar_por_confianza(result.boxes)
        
        if len(detecciones_validas) == 0:
            return self._generar_respuesta_fallo(
                f"Ninguna detección supera el umbral de confianza "
                f"({self.confianza_minima*100:.0f}%)"
            )
        
        # Calcular métricas
        num_filas = len(detecciones_validas)
        desviacion_mm = self._calcular_desviacion(
            detecciones_validas, 
            ancho_imagen
        )
        
        # Generar metadata
        confianzas = [d['confianza'] for d in detecciones_validas]
        
        return {
            'success': True,
            'filas': num_filas,
            'desviacion_mm': desviacion_mm,
            'metadata': {
                'total_detectado': len(result.boxes),
                'detecciones_validas': len(detecciones_validas),
                'confianza_promedio': float(np.mean(confianzas)),
                'confianza_minima': float(np.min(confianzas)),
                'confianza_maxima': float(np.max(confianzas)),
                'ancho_imagen': ancho_imagen,
                'alto_imagen': alto_imagen
            }
        }
    
    def _filtrar_por_confianza(self, boxes) -> List[Dict]:
        """
        Filtra cajas de detección por umbral de confianza.
        
        Args:
            boxes: Objeto boxes de YOLO results
            
        Returns:
            Lista de diccionarios con datos de detecciones válidas
        """
        detecciones = []
        
        for box in boxes:
            confianza = float(box.conf[0].item())
            
            if confianza >= self.confianza_minima:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detecciones.append({
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2,
                    'ancho': x2 - x1,
                    'alto': y2 - y1,
                    'confianza': confianza,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return detecciones
    
    def _calcular_desviacion(self, 
                            detecciones: List[Dict], 
                            ancho_imagen: int) -> float:
        """
        Calcula la desviación en milímetros.
        
        Estrategia: Encontrar el objeto más cercano al punto de referencia.
        
        Args:
            detecciones: Lista de diccionarios con detecciones
            ancho_imagen: Ancho de la imagen en píxeles
            
        Returns:
            Desviación en mm (positivo=derecha, negativo=izquierda)
        """
        # Determinar punto de referencia
        if self.usar_centro_imagen or self.referencia_x_custom is None:
            punto_referencia = ancho_imagen / 2
        else:
            punto_referencia = self.referencia_x_custom
        
        # Buscar objeto más cercano al punto de referencia
        min_distancia_abs = float('inf')
        desviacion_objetivo = 0.0
        
        for det in detecciones:
            desviacion_px = det['center_x'] - punto_referencia
            distancia_abs = abs(desviacion_px)
            
            if distancia_abs < min_distancia_abs:
                min_distancia_abs = distancia_abs
                desviacion_objetivo = desviacion_px
        
        # Convertir a mm
        desviacion_mm = desviacion_objetivo * self.mm_per_pixel
        
        return desviacion_mm
    
    def _generar_respuesta_fallo(self, razon: str) -> Dict:
        """
        Genera respuesta estructurada para casos de fallo.
        
        Args:
            razon: Descripción del fallo
            
        Returns:
            Dict con success=False y metadata
        """
        print(f"⚠️ Procesamiento fallido: {razon}")
        
        return {
            'success': False,
            'filas': 0,
            'desviacion_mm': 0.0,
            'metadata': {
                'razon_fallo': razon
            }
        }
    
    def validar_resultado(self, resultado: Dict) -> Tuple[bool, List[str]]:
        """
        Valida un resultado antes de enviarlo al PLC.
        
        Validaciones:
        - Rango razonable de desviación
        - Número de filas lógico
        - Coherencia de datos
        
        Args:
            resultado: Diccionario retornado por procesar_resultados()
            
        Returns:
            (es_valido, lista_de_advertencias)
        """
        advertencias = []
        
        if not resultado['success']:
            return True, []  # Los fallos no necesitan validación adicional
        
        # Validar desviación
        desv = abs(resultado['desviacion_mm'])
        if desv > 500:  # >50cm es sospechoso
            advertencias.append(
                f"⚠️ Desviación muy grande: {resultado['desviacion_mm']:.2f}mm"
            )
        
        # Validar número de filas
        filas = resultado['filas']
        if filas < 0 or filas > 100:
            advertencias.append(f"⚠️ Número de filas inusual: {filas}")
        
        # Coherencia
        if filas == 0 and desv > 0:
            advertencias.append("⚠️ Incoherencia: 0 filas pero desviación != 0")
        
        return len(advertencias) == 0, advertencias
    
    def ajustar_calibracion(self, nuevo_mm_per_pixel: float) -> None:
        """
        Ajusta la calibración espacial del sistema.
        
        Args:
            nuevo_mm_per_pixel: Nueva relación mm/píxel
        """
        self.mm_per_pixel = nuevo_mm_per_pixel
        print(f"🔧 Calibración actualizada: {self.mm_per_pixel} mm/píxel")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Configuración de ejemplo
    config_ejemplo = {
        'vision': {
            'mm_per_pixel': 0.5,
            'confianza_minima': 0.5,
            'usar_centro_imagen': True,
            'referencia_x_custom': None
        }
    }
    
    processor = VisionProcessor(config_ejemplo)
    print("✅ VisionProcessor inicializado")