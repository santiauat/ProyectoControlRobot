"""
Sistema de logging para debugging y análisis
"""

import logging
from datetime import datetime
from typing import Dict
import os


def setup_logger(nombre: str = 'PLCSystem', 
                nivel: int = logging.INFO,
                archivo_log: str = None) -> logging.Logger:
    """
    Configura el sistema de logging.
    
    Args:
        nombre: Nombre del logger
        nivel: Nivel de logging (INFO, DEBUG, etc.)
        archivo_log: Ruta opcional para guardar logs en archivo
        
    Returns:
        Objeto Logger configurado
    """
    logger = logging.getLogger(nombre)
    logger.setLevel(nivel)
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    # Formato de mensajes
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (opcional)
    if archivo_log:
        os.makedirs(os.path.dirname(archivo_log) if os.path.dirname(archivo_log) else '.', 
                   exist_ok=True)
        file_handler = logging.FileHandler(archivo_log, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_resultado_procesamiento(resultado: Dict, logger: logging.Logger = None):
    """
    Registra un resultado de procesamiento de forma estructurada.
    
    Args:
        resultado: Diccionario con el resultado del procesamiento
        logger: Logger a usar (si es None, usa print)
    """
    separador = "=" * 70
    
    mensaje = f"\n{separador}\n"
    mensaje += "RESULTADO DE PROCESAMIENTO YOLO → PLC\n"
    mensaje += f"{separador}\n"
    
    if resultado['success']:
        mensaje += f"✅ Estado: ÉXITO\n"
        mensaje += f"📊 Filas detectadas: {resultado['filas']}\n"
        mensaje += f"📏 Desviación: {resultado['desviacion_mm']:.2f} mm\n"
        
        if 'metadata' in resultado:
            meta = resultado['metadata']
            mensaje += f"\n📈 Metadata:\n"
            mensaje += f"   • Detecciones totales: {meta.get('total_detectado', 'N/A')}\n"
            mensaje += f"   • Detecciones válidas: {meta.get('detecciones_validas', 'N/A')}\n"
            mensaje += f"   • Confianza promedio: {meta.get('confianza_promedio', 0):.2%}\n"
    else:
        mensaje += f"❌ Estado: FALLO\n"
        mensaje += f"📋 Razón: {resultado.get('metadata', {}).get('razon_fallo', 'Desconocida')}\n"
    
    mensaje += f"{separador}\n"
    
    if logger:
        logger.info(mensaje)
    else:
        print(mensaje)


def log_estado_plc(estado: Dict, logger: logging.Logger = None):
    """
    Registra el estado del PLC de forma legible.
    
    Args:
        estado: Diccionario con estado del PLC
        logger: Logger a usar
    """
    if not estado.get('conectado'):
        mensaje = "❌ PLC desconectado"
    else:
        mensaje = f"PLC Estado → Trigger: {estado.get('descripcion_trigger')}, "
        mensaje += f"Filas: {estado.get('filas', 0)}"
    
    if logger:
        logger.info(mensaje)
    else:
        print(mensaje)