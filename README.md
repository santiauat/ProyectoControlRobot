# Sistema PLC-YOLO para Organización de Tubos

Sistema integrado de visión artificial y control PLC para automatización robótica.

## 🎯 Características

- ✅ Comunicación con PLC Mitsubishi (MC Protocol Type3E)
- ✅ Detección de objetos con YOLO
- ✅ Cálculo automático de desviación y conteo
- ✅ Interfaz gráfica intuitiva
- ✅ Modo simulación para pruebas sin hardware
- ✅ Logging detallado para debugging

## 📋 Requisitos

- Python 3.8+
- PLC Mitsubishi con soporte MC Protocol
- Cámara USB o IP
- Modelo YOLO entrenado (.pt)

## 🚀 Instalación

1. Clonar/descargar el proyecto

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar `config/plc_config.json` con tu IP de PLC

4. Ejecutar:
```bash
python main.py
```

## 📖 Uso

### Configuración Inicial

1. **Conectar PLC**: Click en "🔌 Conectar PLC"
2. **Cargar Modelo**: Seleccionar archivo `.pt` de YOLO
3. **Abrir Cámara**: Iniciar captura de video
4. **Iniciar Sistema**: Click en "▶️ INICIAR SISTEMA"

### Modo Simulación

Para probar sin PLC físico:
- Marcar "Modo Simulación (sin PLC)"
- El sistema procesará frames continuamente sin esperar D28=99

### Calibración

Editar en `config/plc_config.json`:
```json
"vision": {
    "mm_per_pixel": 0.5,  // ← Ajustar según tu setup
    "confianza_minima": 0.5
}
```

## 🔧 Protocolo PLC

### Handshake
1. PLC escribe `99` en D28 → "Necesito inspección"
2. Python procesa imagen
3. Python escribe:
   - D29: Desviación (mm × 100, int32)
   - D14: Número de filas (int16)
   - D28: `88` (éxito) o `77` (error)

### Códigos de Estado (D28)
- `99` = Solicitud de inspección
- `88` = Inspección exitosa
- `77` = Error en inspección
- `0` = Sistema idle

## 📁 Estructura del Proyecto