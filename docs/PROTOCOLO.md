# Protocolo de Comunicación PLC

## Resumen

Protocolo: **Mitsubishi MC Protocol Type3E**
Transporte: **TCP/IP**
Puerto: **5007**

## Direcciones de Memoria

| Dirección | Tipo | Uso |
|-----------|------|-----|
| D28 | Word (16b) | Canal de control bidireccional |
| D29-D30 | DWord (32b) | Desviación en 1/100 mm |
| D14 | Word (16b) | Número de filas |

## Códigos de Estado (D28)

- `99`: PLC solicita inspección
- `88`: Python reporta éxito
- `77`: Python reporta error
- `0`: Sistema en idle

## Secuencia de Comunicación