"""
PLCController - Módulo de comunicación con PLC Mitsubishi
Implementa el protocolo MC Type3E y el handshake de control
"""

import pymcprotocol
import json
from typing import Optional, Dict, Tuple


class PLCController:
    """
    Controlador para comunicación con PLC Mitsubishi via MC Protocol.
    
    Responsabilidades:
    - Gestionar conexión TCP/IP con el PLC
    - Implementar protocolo de handshake (D28: 99→88/77)
    - Codificar/decodificar datos (mm → int32, etc.)
    - Manejar reconexiones automáticas
    """
    
    def __init__(self, config_file: str = 'config/plc_config.json'):
        """
        Inicializa el controlador con configuración desde JSON.
        
        Args:
            config_file: Ruta al archivo de configuración
        """
        self.config = self._cargar_configuracion(config_file)
        self.mc = None
        self.is_connected = False
        
        # Extraer configuraciones
        conn = self.config['conexion']
        dirs = self.config['direcciones']
        codigos = self.config['codigos_estado']
        
        self.ip_plc = conn['ip_plc']
        self.puerto_plc = conn['puerto_plc']
        
        self.DEV_TRIGGER = dirs['dispositivo_trigger']
        self.DEV_RESULTADO_VALOR = dirs['dispositivo_valor']
        self.DEV_RESULTADO_FILAS = dirs['dispositivo_filas']
        
        self.VAL_SOLICITUD = codigos['valor_solicitud']
        self.VAL_EXITO = codigos['valor_exito']
        self.VAL_ERROR = codigos['valor_error']
    
    def _cargar_configuracion(self, config_file: str) -> Dict:
        """Carga configuración desde archivo JSON"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuración cargada desde {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Archivo {config_file} no encontrado, usando valores por defecto")
            return self._configuracion_por_defecto()
        except json.JSONDecodeError as e:
            print(f"❌ Error parseando JSON: {e}")
            raise
    
    def _configuracion_por_defecto(self) -> Dict:
        """Retorna configuración por defecto si falla la carga"""
        return {
            "conexion": {
                "ip_plc": "192.168.100.120",
                "puerto_plc": 5007
            },
            "direcciones": {
                "dispositivo_trigger": "D28",
                "dispositivo_valor": "D29",
                "dispositivo_filas": "D14"
            },
            "codigos_estado": {
                "valor_solicitud": 99,
                "valor_exito": 88,
                "valor_error": 77
            }
        }
    
    def conectar(self) -> bool:
        """
        Establece conexión con el PLC.
        
        Returns:
            True si la conexión fue exitosa, False en caso contrario
        """
        print(f"🔌 Conectando al PLC en {self.ip_plc}:{self.puerto_plc}...")
        try:
            self.mc = pymcprotocol.Type3E()
            self.mc.connect(self.ip_plc, self.puerto_plc)
            self.is_connected = True
            print("✅ Conexión PLC establecida exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error al conectar con PLC: {e}")
            self.is_connected = False
            return False
    
    def desconectar(self) -> None:
        """Cierra la conexión con el PLC de forma segura"""
        if self.is_connected and self.mc:
            try:
                self.mc.close()
                print("✅ Desconectado del PLC")
            except Exception as e:
                print(f"⚠️ Error al desconectar: {e}")
            finally:
                self.is_connected = False
                self.mc = None
    
    def leer_solicitud_inspeccion(self) -> bool:
        """
        Lee el registro D28 para verificar si hay solicitud de inspección.
        
        Protocolo:
        - D28 = 99: PLC solicita inspección
        - Otros valores: No hay solicitud
        
        Returns:
            True si D28 == 99, False en caso contrario
        """
        if not self.is_connected:
            return False
        
        try:
            valor = self.mc.batchread_wordunits(
                headdevice=self.DEV_TRIGGER, 
                readsize=1
            )[0]
            
            if valor == self.VAL_SOLICITUD:
                print(f"📥 Solicitud de inspección detectada ({self.DEV_TRIGGER}={self.VAL_SOLICITUD})")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error al leer {self.DEV_TRIGGER}: {e}")
            self.is_connected = False
            return False
    
    def escribir_resultados(self, 
                          desviacion_mm: float, 
                          num_filas: int, 
                          exito: bool) -> bool:
        """
        Escribe los resultados de la inspección al PLC.
        
        Protocolo de escritura (orden crítico):
        1. D29 (desviación en 1/100 mm, 32 bits)
        2. D14 (número de filas, 16 bits)
        3. D28 (estado: 88=éxito, 77=error)
        
        Args:
            desviacion_mm: Desviación en milímetros (float)
            num_filas: Número de filas detectadas (int)
            exito: True si la detección fue exitosa
            
        Returns:
            True si la escritura fue exitosa
        """
        if not self.is_connected:
            print("❌ No se puede escribir: sin conexión PLC")
            return False
        
        try:
            if exito:
                # Convertir desviación a formato PLC (1/100 mm)
                valor_desviacion = int(round(desviacion_mm * 100.0))
                palabras_valor = self._int32_to_words(valor_desviacion)
                
                # Validar número de filas
                valor_filas = max(0, int(num_filas))
                
                # ORDEN CRÍTICO: Datos primero, estado después
                self.mc.batchwrite_wordunits(
                    headdevice=self.DEV_RESULTADO_VALOR, 
                    values=palabras_valor
                )
                self.mc.batchwrite_wordunits(
                    headdevice=self.DEV_RESULTADO_FILAS, 
                    values=[valor_filas]
                )
                self.mc.batchwrite_wordunits(
                    headdevice=self.DEV_TRIGGER, 
                    values=[self.VAL_EXITO]
                )
                
                print(f"✅ Resultados enviados: Desv={desviacion_mm:.2f}mm ({valor_desviacion}), "
                      f"Filas={valor_filas}, Estado=ÉXITO({self.VAL_EXITO})")
            else:
                # Error: enviar ceros
                palabras_cero = self._int32_to_words(0)
                
                self.mc.batchwrite_wordunits(
                    headdevice=self.DEV_RESULTADO_VALOR, 
                    values=palabras_cero
                )
                self.mc.batchwrite_wordunits(
                    headdevice=self.DEV_RESULTADO_FILAS, 
                    values=[0]
                )
                self.mc.batchwrite_wordunits(
                    headdevice=self.DEV_TRIGGER, 
                    values=[self.VAL_ERROR]
                )
                
                print(f"❌ Error enviado al PLC: Estado=ERROR({self.VAL_ERROR})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error al escribir resultados: {e}")
            self.is_connected = False
            return False
    
    def _int32_to_words(self, n: int) -> list:
        """
        Convierte un entero con signo de 32 bits a dos palabras de 16 bits.
        
        Formato PLC: [low_word, high_word]
        Ejemplo: -1250 → [0xFB1E, 0xFFFF]
        
        Args:
            n: Entero con signo (-2147483648 a 2147483647)
            
        Returns:
            Lista [low_word, high_word]
        """
        # Clamp al rango int32
        n = max(-2147483648, min(n, 2147483647))
        
        # Convertir a unsigned si es negativo
        if n < 0:
            n = n + (1 << 32)
        
        low_word = n & 0xFFFF
        high_word = (n >> 16) & 0xFFFF
        
        return [low_word, high_word]
    
    def verificar_conexion(self) -> bool:
        """
        Verifica si la conexión con el PLC sigue activa.
        
        Returns:
            True si la conexión está activa
        """
        if not self.is_connected or not self.mc:
            return False
        
        try:
            # Intenta leer el registro de trigger
            self.mc.batchread_wordunits(headdevice=self.DEV_TRIGGER, readsize=1)
            return True
        except Exception:
            self.is_connected = False
            return False
    
    def obtener_estado_sistema(self) -> Dict:
        """
        Lee el estado completo del sistema desde el PLC.
        
        Returns:
            Diccionario con estado actual de D28, D29, D14
        """
        if not self.is_connected:
            return {'conectado': False}
        
        try:
            trigger = self.mc.batchread_wordunits(headdevice=self.DEV_TRIGGER, readsize=1)[0]
            filas = self.mc.batchread_wordunits(headdevice=self.DEV_RESULTADO_FILAS, readsize=1)[0]
            
            return {
                'conectado': True,
                'trigger': trigger,
                'filas': filas,
                'descripcion_trigger': self._describir_codigo(trigger)
            }
        except Exception as e:
            print(f"⚠️ Error leyendo estado: {e}")
            return {'conectado': False, 'error': str(e)}
    
    def _describir_codigo(self, codigo: int) -> str:
        """Convierte código numérico a descripción legible"""
        if codigo == self.VAL_SOLICITUD:
            return "SOLICITUD PENDIENTE"
        elif codigo == self.VAL_EXITO:
            return "ÚLTIMA INSPECCIÓN: ÉXITO"
        elif codigo == self.VAL_ERROR:
            return "ÚLTIMA INSPECCIÓN: ERROR"
        elif codigo == 0:
            return "IDLE"
        else:
            return f"CÓDIGO DESCONOCIDO ({codigo})"


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Test básico del controlador
    plc = PLCController()
    
    if plc.conectar():
        print("\n📊 Estado del sistema:")
        estado = plc.obtener_estado_sistema()
        print(estado)
        
        plc.desconectar()