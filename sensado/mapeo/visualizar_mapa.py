"""Dibuja el mapa XML de árboles como SVG sin instalar bibliotecas extra.

Uso: python3 visualizar_mapa.py
Lee ~/robot_ws/mapa_arboles.xml y genera ~/robot_ws/mapa_arboles.svg.
"""

from pathlib import Path
import html
import math
import xml.etree.ElementTree as ET


# ============================================================
# 1. RUTAS Y LECTURA DEL MAPA
# ============================================================
directorio = Path(__file__).resolve().parent

archivo_xml = directorio / 'mapa_arboles.xml'
archivo_svg = directorio / 'mapa_arboles.svg'

raiz = ET.parse(archivo_xml).getroot()
puntos = [
    (float(e.attrib['x']), float(e.attrib['y']))
    for e in raiz.iter()
    if e.tag.split('}')[-1] == 'point' and 'x' in e.attrib and 'y' in e.attrib
]
if not puntos:
    raise SystemExit('El XML no contiene árboles con coordenadas x/y.')

# ============================================================
# 2. ESCALA DEL PLANO XY
# ============================================================
ancho, alto = 900, 540
margen = 80
min_x = min(x for x, y in puntos) - 1.0
max_x = max(x for x, y in puntos) + 1.0
min_y = min(y for x, y in puntos) - 1.0
max_y = max(y for x, y in puntos) + 1.0
escala = min((ancho - 2*margen)/(max_x-min_x),
             (alto - 2*margen)/(max_y-min_y))
centro_x = (min_x + max_x)/2
centro_y = (min_y + max_y)/2

def convertir(x, y):
    return (ancho/2 + (x-centro_x)*escala,
            alto/2 - (y-centro_y)*escala)

# ============================================================
# 3. GENERAR LA VISUALIZACIÓN SVG
# ============================================================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">']
svg += [
    '<rect width="100%" height="100%" fill="#f8fafc"/>',
    '<text x="45" y="42" font-family="Arial" font-size="25" font-weight="bold" fill="#132b42">Mapa de árboles</text>',
    '<text x="45" y="66" font-family="Arial" font-size="14" fill="#52657a">Coordenadas detectadas por LiDAR · Marco odom · Unidades: m</text>',
]

# Retícula cada metro (sin agrupar árboles en una cuadrícula arbitraria)
for x in range(math.floor(min_x), math.ceil(max_x)+1):
    sx, _ = convertir(x, centro_y)
    svg.append(f'<line x1="{sx:.2f}" y1="100" x2="{sx:.2f}" y2="478" stroke="#dbe3ec" stroke-width="1"/>')
    svg.append(f'<text x="{sx:.2f}" y="497" text-anchor="middle" font-family="Arial" font-size="12" fill="#597084">{x}</text>')
for y in range(math.floor(min_y), math.ceil(max_y)+1):
    _, sy = convertir(centro_x, y)
    if 100 <= sy <= 478:
        svg.append(f'<line x1="40" y1="{sy:.2f}" x2="855" y2="{sy:.2f}" stroke="#dbe3ec" stroke-width="1"/>')
        svg.append(f'<text x="22" y="{sy+4:.2f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#597084">{y}</text>')

# Centros de los árboles
for i, (x, y) in enumerate(puntos, 1):
    sx, sy = convertir(x, y)
    etiqueta = html.escape(f'Árbol {i}')
    svg += [
        f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="13" fill="#16835b" stroke="#075238" stroke-width="3"/>',
        f'<text x="{sx:.2f}" y="{sy-24:.2f}" font-family="Arial" text-anchor="middle" font-size="14" font-weight="bold" fill="#173b30">{etiqueta}</text>',
        f'<text x="{sx:.2f}" y="{sy+34:.2f}" font-family="Arial" text-anchor="middle" font-size="12" fill="#34485a">({x:.3f}, {y:.3f})</text>',
    ]

svg += [
    '<text x="450" y="528" text-anchor="middle" font-family="Arial" font-size="14" fill="#52657a">Eje X (m)</text>',
    '<text x="882" y="290" text-anchor="middle" font-family="Arial" font-size="14" fill="#52657a" transform="rotate(-90, 882, 290)">Eje Y (m)</text>',
    '</svg>',
]
archivo_svg.write_text('\n'.join(svg), encoding='utf-8')
print(f'Visualización creada: {archivo_svg}')
