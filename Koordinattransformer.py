#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:39:52 2024

@author: Torjus

Koordinattransformer
"""
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from matplotlib.ticker import ScalarFormatter

# Definer fastpunktene med originale x-, y- og z-koordinater i NGO 1948 akse 1 system
points = {
    'a': (-15581.866, 56416.856, 10),
    'b': (-15586.662, 56379.423, 10),
    'c': (-15576.459, 56255.143, 10),
    'd': (-15510.858, 56036.327, 8),
    'e': (-15504.371, 56024.819, 8),
    'f': (-15443.652, 55957.687, 7)
}



# Funksjon for å generere en sirkelbue med spesifisert radius mellom to punkter
def arc_between_points(p1, p2, radius, z1, z2, num_points=100):
    # Beregn midtpunktet mellom p1 og p2
    midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    # Beregn vektor fra p1 til p2 og lengden på denne
    vec_p1_p2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    dist_p1_p2 = np.linalg.norm(vec_p1_p2)

    # Juster radius hvis den er mindre enn halvparten av avstanden mellom punktene
    if radius < dist_p1_p2 / 2:
        radius = dist_p1_p2 / 2

    # Beregn vinkel for sirkelsegmentet basert på radius
    angle = 2 * np.arcsin(dist_p1_p2 / (2 * radius))

    # Finn senterpunktet for sirkelen som skal inneholde buen
    perp_vec = np.array([-vec_p1_p2[1], vec_p1_p2[0]]) / dist_p1_p2  # Perpendikulær vektor
    center = midpoint + np.sqrt(radius**2 - (dist_p1_p2 / 2)**2) * perp_vec

    # Generer punkter langs buen
    theta_start = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    theta_end = theta_start + angle
    theta = np.linspace(theta_start, theta_end, num_points)

    x_arc = center[0] + radius * np.cos(theta)
    y_arc = center[1] + radius * np.sin(theta)
    z_arc = np.linspace(z1, z2, num_points)  # Lineær interpolasjon for z-verdier

    return x_arc, y_arc, z_arc

# Angi radius for kurvene
radius = 300
radius2 = 30

# Generer og samle alle segmentene i riktig rekkefølge
full_line_points = []

# Legg til linjestykke mellom a og b
full_line_points.append([points['a'][0], points['a'][1], points['a'][2]])
full_line_points.append([points['b'][0], points['b'][1], points['b'][2]])

# Kurve mellom b og c
bc_x, bc_y, bc_z = arc_between_points(points['b'], points['c'], radius, points['b'][2], points['c'][2])
full_line_points.extend([[x, y, z] for x, y, z in zip(bc_x, bc_y, bc_z)])

# Linjestykke mellom c og d
full_line_points.append([points['c'][0], points['c'][1], points['c'][2]])
full_line_points.append([points['d'][0], points['d'][1], points['d'][2]])

# Kurve mellom d og e
de_x, de_y, de_z = arc_between_points(points['d'], points['e'], radius2, points['d'][2], points['e'][2])
full_line_points.extend([[x, y, z] for x, y, z in zip(de_x, de_y, de_z)])

# Linjestykke mellom e og f
full_line_points.append([points['e'][0], points['e'][1], points['e'][2]])
full_line_points.append([points['f'][0], points['f'][1], points['f'][2]])

# Snu listen opp ned
full_line_points.reverse()

# Definer transformasjonen fra NGO 1948 akse 1 til UTM 32 ETRF89
transformer = Transformer.from_crs("EPSG:27391", "EPSG:25832", always_xy=True)

# Transformere alle punktene fra NGO 1948 akse 1 til UTM 32 ETRF89
full_line_points_utm32 = [list(transformer.transform(x, y)) + [z] for x, y, z in full_line_points]

# Manuell translasjon i x- og y-retning etter transformasjonen
x_translation = - 312689.508  # Angi ønsket x-translasjon
y_translation = - 6489686.253   # Angi ønsket y-translasjon
translated_points_utm32 = [[x + x_translation, y + y_translation, z] for x, y, z in full_line_points_utm32]

# Skriv ut resultatene
print("Transformerte og translaterte punkter i UTM 32 ETRF89:")
for point in translated_points_utm32:
    print(point)

# Plot linjen i UTM 32 ETRF89 uten z (2D-visning)
plt.plot([p[0] for p in translated_points_utm32], [p[1] for p in translated_points_utm32], 'b')
plt.xlabel("Øst (UTM 32 ETRF89)")
plt.ylabel("Nord (UTM 32 ETRF89)")
plt.title("Full polylinje i UTM 32 ETRF89 med translasjon")
plt.axis("equal")

plt.grid()
plt.show()

# Skriv punktene til en .xyz-fil i formatet x y z
with open("line_original_coordinates.xyz", "w") as file:
    for point in full_line_points_utm32:
        # Formater hvert punkt som x y z og skriv det til filen
        file.write(f"{point[0]} {point[1]} {point[2]}\n")

# Skriv punktene til en .xyz-fil i formatet x y z
with open("line_translated_coordinates.xyz", "w") as file:
    for point in translated_points_utm32:
        # Formater hvert punkt som x y z og skriv det til filen
        file.write(f"{point[0]} {point[1]} {point[2]}\n")


















