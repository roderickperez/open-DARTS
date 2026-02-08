import math
import matplotlib.pyplot as plt

def generar_linea_angular(base_linea, angulo, longitud=100):
    """
    Genera una línea que parte del inicio de la base y forma un ángulo con ella.
    base_linea: (x1, y1, x2, y2)
    angulo: en grados (0-180)
    longitud: longitud de la nueva línea
    """
    x1, y1, x2, y2 = base_linea
    # Calculamos la dirección de la línea base
    base_ang = math.atan2(y2 - y1, x2 - x1)
    
    # Convertimos el ángulo deseado a radianes
    ang_rad = math.radians(angulo)
    
    # Calculamos la dirección absoluta de la nueva línea
    nueva_dir = base_ang + ang_rad
    
    # Calculamos el punto final de la nueva línea
    x2_nueva = x1 + longitud * math.cos(nueva_dir)
    y2_nueva = y1 + longitud * math.sin(nueva_dir)
    
    return (x1, y1, x2_nueva, y2_nueva)

# Línea base
linea_base = (0, 0, 100, 0)

# Ángulos de intersección deseados
angulos = [90, 45, 30, 15]

# Graficamos
plt.figure(figsize=(6,6))
plt.plot([linea_base[0], linea_base[2]], [linea_base[1], linea_base[3]], 'k-', label='Línea base (0°)')

for ang in angulos:
    x1, y1, x2, y2 = generar_linea_angular(linea_base, ang)
    plt.plot([x1, x2], [y1, y2], label=f'{ang}°')

plt.axis('equal')
plt.legend()
plt.grid(True)
plt.title("Líneas con diferentes ángulos de intersección")
plt.show()
