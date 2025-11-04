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


# <<< FUNCIÓN CORREGIDA >>>
def log_resultado_procesamiento(resultado: Dict, logger: logging.Logger = None):
    """
    Registra un resultado de procesamiento DUAL de forma estructurada.
    
    Args:
        resultado: Diccionario con el resultado del procesamiento
        logger: Logger a usar (si es None, usa print)
    """
    separador = "=" * 70
    
    mensaje = f"\n{separador}\n"
    mensaje += "RESULTADO DE PROCESAMIENTO DUAL → PLC\n"
    mensaje += f"{separador}\n"
    
    # <<< CORRECCIÓN: Leer 'codigo_respuesta_plc' en lugar de 'success' o 'metadata' >>>
    codigo_plc = resultado.get('codigo_respuesta_plc', -1)
    
    if codigo_plc == 2: # PARADA
        mensaje += f"🛑 Estado: PARADA CRÍTICA (Código {codigo_plc})\n"
        mensaje += f"📋 Razón: {resultado.get('log_z', 'Error lateral desconocido')}\n"
    elif codigo_plc == 1: # FALLO QC
        mensaje += f"⚠️ Estado: FALLO QC (Código {codigo_plc})\n"
    else: # OK
        mensaje += f"✅ Estado: ÉXITO (Código {codigo_plc})\n"

    mensaje += "\n--- DATOS ENVIADOS A PLC ---\n"
    mensaje += f"  • Éxito (a D28): {resultado.get('plc_success', False)}\n"
    mensaje += f"  • Filas (a D14): {resultado.get('filas', 0)}\n"
    mensaje += f"  • Desviación Z (a D29): {resultado.get('desviacion_y_mm', 0.0):.2f} mm\n"

    mensaje += "\n--- DATOS DE DIAGNÓSTICO ---\n"
    mensaje += f"  • Corrección Z (cálculo): {resultado.get('correccion_z_cmm', 0)} cMM\n"
    mensaje += f"  • Desviación Y (cálculo): {resultado.get('desviacion_y_px', 0)} px\n"
    mensaje += f"  • Log Lateral (Z): {resultado.get('log_z', 'N/A')}\n"
    
    mensaje += f"{separador}\n"
    
    if logger:
        if codigo_plc == 2:
            logger.error(mensaje) # Loguear paradas como ERROR
        else:
            logger.info(mensaje)
    else:
        print(mensaje)


def log_estado_plc(controlador_plc, logger: logging.Logger, solicitud_detectada: bool):
    """
    Registra el estado del PLC de forma legible.
    
    Args:
        controlador_plc: Objeto PLCController
        logger: Logger a usar
        solicitud_detectada: bool
    """
    if not controlador_plc or not controlador_plc.is_connected:
        mensaje = "❌ PLC desconectado"
        if logger: logger.warning(mensaje)
        else: print(mensaje)
        return

    try:
        if solicitud_detectada:
            mensaje = f"🟢 PLC Estado: SOLICITUD RECIBIDA (D28={controlador_plc.VAL_SOLICITUD})"
        else:
            # Leer el valor actual para saber por qué no está listo (opcional)
            # valor_actual = controlador_plc.mc.batchread_wordunits(headdevice=controlador_plc.DEV_TRIGGER, readsize=1)[0]
            # mensaje = f"⚪ PLC Estado: Esperando (D28={valor_actual})"
            
            # Mensaje simple para no saturar el log
            mensaje = f"⚪ PLC Estado: Esperando solicitud (D28 != {controlador_plc.VAL_SOLICITUD})"
        
        if logger:
            logger.debug(mensaje) # Usar DEBUG para no saturar el log de INFO
        else:
            print(mensaje)
            
    except Exception as e:
        mensaje = f"⚠️ Error leyendo estado de PLC: {e}"
        if logger: logger.error(mensaje)
        else: print(mensaje)