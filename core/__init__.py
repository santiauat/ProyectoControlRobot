"""
Módulo core del sistema PLC-YOLO
Contiene la lógica principal de comunicación y procesamiento
"""

from .plc_controller import PLCController
from .vision_processor import VisionProcessor

__all__ = ['PLCController', 'VisionProcessor']