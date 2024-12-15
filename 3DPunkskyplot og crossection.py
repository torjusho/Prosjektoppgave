#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:49:24 2024

@author: Torjus

Origo i input-fil er translert slik at det ligger på tuppen av molohodet. 
Dette for å gjøre senere beregninger og smudere og for å få et mer sømløst resultat.

Ved bruk av utranslerte koordinater må koden få noen endringer

3D-plot fungerer dårlig for høy oppløsning av punkter
"""

import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############## Linjepunkter ###############
file_path_line = '/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/3Dmodifisert/line_translated_coordinates.xyz'
center_line  = np.genfromtxt(file_path_line)

''' 
Funksjonen returnerer punkt-koordinat for senter av tverrsnittskube som ligger på senterlinje i moloen. 
Senterlinjens vinkel til y-aksen blir også returnert
'''

def point_and_angle_at_line(line_points, length_from_breakwater_head):
    # Beregn forskjeller mellom hver suksessive par av punkter i x- og y-retning
    delta = np.diff(line_points[:, :2], axis=0)
    segment_lengths = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
    
    cumulative_length = 0
    for i in range(len(segment_lengths)):
        cumulative_length += segment_lengths[i]
        if cumulative_length >= length_from_breakwater_head:
            # Interpoler mellom punktene for å finne eksakt posisjon
            overshoot = cumulative_length - length_from_breakwater_head
            ratio = (segment_lengths[i] - overshoot) / segment_lengths[i]
            x = line_points[i, 0] + ratio * (line_points[i+1, 0] - line_points[i, 0])
            y = line_points[i, 1] + ratio * (line_points[i+1, 1] - line_points[i, 1])
            z = 0  # eller bruk interpolering for Z, hvis ønsket

            # Beregn vinkelen til y-aksen
            dx = line_points[i+1, 0] - line_points[i, 0]
            dy = line_points[i+1, 1] - line_points[i, 1]
            angle_to_y_axis = np.arctan2(dx, dy)  # Vinkel i radianer mellom linjen og y-aksen
            
            return x, y, z, angle_to_y_axis
    
    # Hvis ønsket lengde overstiger total lengde, returner siste punkt og vinkel
    x, y, z = line_points[-1, 0], line_points[-1, 1], 5  # Evt. bruk line_points[-1, 2] for z
    dx = line_points[-1, 0] - line_points[-2, 0]
    dy = line_points[-1, 1] - line_points[-2, 1]
    angle_to_y_axis = np.arctan2(dx, dy)
    
    return x, y, z, angle_to_y_axis


''' Distanse fra molohode, langs senterlinje, til senter for tverrsnittskube
    Geometriske mål på kuben
    Koordinater for punkt langs linje i angitt avstand, og linjens vinkel til y-aksen, som kuben skal roteres med'''
########################## Inputs ###################################
length_from_breakwater_head = 150  # Lengden fra molohodet, målt langs senterlinjen i moloen
x_size , y_size, z_size = 100, 14, 25 # Kube for å trekke ut tverrsnitt Kun positive verdier her
center_x, center_y, center_z, rotation_angle = point_and_angle_at_line(center_line, length_from_breakwater_head) # Posisjon og rotasjon

########################## U-ferdig del ######################
crossections = {"A": [], 
                "B": [], 
                "C": [], 
                "D": [253.35], 
                "E": [], 
                "F": [], 
                "G": [], 
                "H": [], 
                "I": [], 
                'J': [],}
################################################################

# Kube for å isolere punkter som er innenfor grensene som er satt
half_x, half_y, half_z = x_size / 2, y_size / 2, z_size / 2
points = np.array([[half_x,  half_y,  half_z],
                      [ half_x, -half_y,  half_z],
                      [ half_x,  half_y, -half_z],
                      [ half_x, -half_y, -half_z],
                      [-half_x,  half_y,  half_z],
                      [-half_x,  half_y, -half_z],
                      [-half_x, -half_y,  half_z],
                      [-half_x, -half_y, -half_z]])
    
# Rotasjonsmatrise for å rotere kuben om z-aksen:
theta = - rotation_angle
# theta = np.radians(rotation_angle) # for manuell inntsating av rotasjon i grader
rotation_matrix = np.array([[np.cos(theta),-np.sin(theta),0],
                            [np.sin(theta) ,np.cos(theta) ,0],
                            [0             ,0             ,1]])
     
#Roterer om origo av kuben
rotated_matrix = points @ rotation_matrix.T

#Translerer matrisen til det aktuelle punktet på linjen
shifted_matrix = rotated_matrix + np.array([center_x, center_y, center_z])
    
#Trekker linjer mellom punktene for å danne en kube
edges = [[0,1],[1,6],[6,4],[4,0], #topp
         [2,3],[3,7],[7,5],[5,2], #bunn
         [0,2],[1,3],[6,7],[4,5]] #sider

#Scatter 3d-objekter for kubene
cube_edges = []
cube_edges2 =[]
for edge in edges:
    line = go.Scatter3d(
        x = [shifted_matrix[edge[0],0], shifted_matrix[edge[1],0]],
        y = [shifted_matrix[edge[0],1], shifted_matrix[edge[1],1]],
        z = [shifted_matrix[edge[0],2], shifted_matrix[edge[1],2]],
    mode = 'lines',
    line = dict(color = 'black', width = 4),
    showlegend=False
    )
    cube_edges.append(line)

for edge in edges:
    line = go.Scatter3d(
        x = [points[edge[0],0], points[edge[1],0]],
        y = [points[edge[0],1], points[edge[1],1]],
        z = [points[edge[0],2], points[edge[1],2]],
    mode = 'lines',
    line = dict(color = 'black', width = 1),
    showlegend=False
    )
    cube_edges2.append(line)
        
cube  = cube_edges
cube2 = cube_edges2

###################### Definerer kubenses maks og min punkter######################
cube_points_x = shifted_matrix[:, 0]
cube_points_y = shifted_matrix[:, 1]
cube_points_z = shifted_matrix[:, 2]

cube_points_x_2 = points[:, 0]
cube_points_y_2 = points[:, 1]
cube_points_z_2 = points[:, 2]

######################## Data fra punktskyen #################################
file_path = '/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/Redusert utgave/20cm_20%.xyz'

data  = np.genfromtxt(file_path)

# Hente X, Y og Z- verdier og RGB fra punksky
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
colors = data[:, 3:6] / 255.0  # Normaliserer RGB verdier til [0, 1]

######### Scatterplot av punktskyen som lastes inn #####################
scatter = go.Scatter3d(
    x=X, y=Y, z=Z,
    mode='markers',
    marker=dict(size=2, color=colors, opacity=0.8),
    name = "Punktsky - Sirevåg Hovedmolo")

########################## Akseforhold og layout på plot ##########################
x_range = max(max(X), max(cube_points_x)) - min(min(X), min(cube_points_x))
y_range = max(max(Y), max(cube_points_y)) - min(min(Y), min(cube_points_y))
z_range = max(max(Z), max(cube_points_z)) - min(min(Z), min(cube_points_z))

# Definerer layout og akseforhold
layout = go.Layout(
    scene=dict(
        xaxis=dict(
            nticks=10, 
            range=[min(min(X), min(cube_points_x), min(cube_points_x_2)), 
                   max(max(X), max(cube_points_x), max(cube_points_x_2))],
            title="X Axis",
            tickfont=dict(size=15),  # Størrelse på tall langs aksen
            titlefont=dict(size=15)  # Størrelse på aksetittelen
        ),
        yaxis=dict(
            nticks=10, 
            range=[min(min(Y), min(cube_points_y), min(cube_points_y_2)), 
                   max(max(Y), max(cube_points_y), max(cube_points_y_2))],
            title="Y Axis",
            tickfont=dict(size=15),  # Størrelse på tall langs aksen
            titlefont=dict(size=15)  # Størrelse på aksetittelen
        ),
        zaxis=dict(
            nticks=10, 
            range=[min(min(Z), min(cube_points_z), min(cube_points_z_2)), 
                   max(max(Z), max(cube_points_z), max(cube_points_z_2))],
            title="Z Axis",
            tickfont=dict(size=12),  # Størrelse på tall langs aksen
            titlefont=dict(size=14)  # Størrelse på aksetittelen
        ),
        aspectmode='manual',  # Spesifiser manuelt
        aspectratio=dict(x=x_range, y=y_range, z=z_range),  # 1:1:1 forhold mellom aksene
        camera=dict(
            projection=dict(type="orthographic"),
            eye=dict(x=0, y=0, z=2),
            # eye=dict(x=2, y=2, z=0.3),  # Økt avstand langs X og Y, redusert langs Z for bredere utsnitt
            up=dict(x=-1, y=0, z=0),         # Beholder Z som "oppover"-retning
            center=dict(x=0, y=0, z=0)      # Sentrum i modellen
        )
    ),
    legend=dict(
        font=dict(size=18),  # Skriftstørrelsen på label-tekstene
        orientation="v",  
        yanchor="top",  
        y=0.35,            #Posisjon på label-tekst
        xanchor="right",    
        x=0.9              #Posisjon på label-tekst   
    )
)



''' Filtrerer her ut punkter som er utenfor grenseverdiene satt av kuben. 
    I første omgang blir det ikke tatt hensyn til rotasjon av kuben, slik at det er max og min x,y og z- verdier som bestemmer
    Andregangsfiltrering skjer etter translering og rotering '''
    
##################### Finne maks og minverdier for førstegangsfiltrering av punkter ###########################
x_min, x_max = np.min(cube_points_x), np.max(cube_points_x)
y_min, y_max = np.min(cube_points_y), np.max(cube_points_y)
z_min, z_max = np.min(cube_points_z), np.max(cube_points_z)

# Definere punkter fra punktskyen som er innenfor maks og min- koordinater av rotert kube
points_within_cube = data[
    (X >= x_min) & (X <= x_max) &
    (Y >= y_min) & (Y <= y_max) &
    (Z >= z_min) & (Z <= z_max)
]

# Flytter punktene til origo
points_within_cube[:, 0] -= center_x
points_within_cube[:, 1] -= center_y
points_within_cube[:, 2] -= center_z

#Roterer translerte punkter med invers rotasjonsmatrise
inverse_rotation_matrix = np.array([[np.cos(-theta),-np.sin(-theta), 0],
                                     [np.sin(-theta),np.cos(-theta) , 0],
                                     [0                      ,0                       , 1]])

rotated_points_within_initial_cube = points_within_cube[:, :3] @ inverse_rotation_matrix.T

# Punktene er nå rotert med motsatt verdi for å filtrere punktene innenfor en kube som står vinkelrett på aksesystemet
# Kuben er nå vinkelrett på aksesystemet, og punktene er rotert, kubens hørnekoordinater blir da grenseverdier for andregangsfiltrering
cube_points_x_2 = x_size / 2 
cube_points_y_2 = y_size / 2
cube_points_z_2 = z_size / 2

x_min, x_max = -cube_points_x_2, cube_points_x_2
y_min, y_max = -cube_points_y_2, cube_points_y_2
z_min, z_max = -cube_points_z_2, cube_points_z_2

# Deler punktene opp i x, y og z verdier 
X2 = rotated_points_within_initial_cube[:, 0]
Y2 = rotated_points_within_initial_cube[:, 1]
Z2 = rotated_points_within_initial_cube[:, 2]

# Filtrerer for andre gang, og fjerner punkter som er utenfor
points_within_cube2 = rotated_points_within_initial_cube[
    (X2 >= x_min) & (X2 <= x_max) &
    (Y2 >= y_min) & (Y2 <= y_max) &
    (Z2 >= z_min) & (Z2 <= z_max)
]

#print(points_within_cube2)

X3 = points_within_cube2[:, 0]
Y3 = points_within_cube2[:, 1]
Z3 = points_within_cube2[:, 2]

# Roterer tilbake, og translerer til original posisjon for å visualisere at riktige punkter er beholdt
points_within_cube3 = points_within_cube2[:, :3] @ rotation_matrix.T

points_within_cube3[:, 0] += center_x
points_within_cube3[:, 1] += center_y
points_within_cube3[:, 2] += center_z

X4 = points_within_cube3[:, 0]
Y4 = points_within_cube3[:, 1]
Z4 = points_within_cube3[:, 2]

#points_within_cube2 = np.hstack([rotated_points_within_initial_cube,points_within_cube[:, 3:]])



''' Plotter kuber, punksky og punkter innenfor tverrsnittet '''
############################# 3D plot ###############################
# Scatter for steg i koden:

# # Steg 1: Transler til origo
# scatter_points_whitin_cube = go.Scatter3d(
#     x=points_within_cube[:, 0]+center_x, y=points_within_cube[:, 1]+center_y, z=points_within_cube[:, 2]+center_z,
#     mode='markers',
#     marker=dict(size=2, color='red', opacity=0.8),
#     name = "Points within cube")

# #Steg 2: Roter om origo med vinkel tverrsnittet har på senterlinje
# scatter_points_whitin_cube = go.Scatter3d(
#     x=X2, y=Y2, z=Z2,
#     mode='markers',
#     marker=dict(size=2, color='red', opacity=0.8),
#     name = "Points within cube")

# Steg 3: Filtrer andre gang når punkter og tverrsnittsgrenser står vinkelrett på aksesystem
scatter_points_whitin_cube = go.Scatter3d(
    x=X3, y=Y3, z=Z3,
    mode='markers',
    marker=dict(size=2, color='red', opacity=0.8),
    name = "Points within cube")

# # Steg 4: Roter tilbake og transler til original posisjon
# scatter_points_whitin_cube = go.Scatter3d(
#     x=X4, y=Y4, z=Z4,
#     mode='markers',
#     marker=dict(size=2, color='red', opacity=0.8),
#     name = "Points within cube")



# Fulstendig 3D- plot
scatter_points_whitin_cube2 = go.Scatter3d(
    x=X3, y=Y3, z=Z3,
    mode='markers',
    marker=dict(size=2, color='red', opacity=0.8),
    name = "Points within cube 2")

scatter_points_whitin_cube3 = go.Scatter3d(
    x=X4, y=Y4, z=Z4,
    mode='markers',
    marker=dict(size=2, color='red', opacity=0.8),
    name = "Skade 1")

X_L, Y_L, Z_L = center_line[:, 0], center_line[:, 1], center_line[:, 2]
linje = go.Scatter3d(
    x=X_L, y=Y_L, z=Z_L,
    mode='lines + markers',
    marker=dict(size=4, color='red', opacity=1),
    name = "Senterlinje")



#Første filtrering
#fig = go.Figure(data=[scatter, scatter_points_whitin_cube] + cube + cube2, layout=layout)


#fig = go.Figure(data=[scatter, scatter_points_whitin_cube2, scatter_points_whitin_cube3] + cube + cube2, layout=layout)
#fig = go.Figure(data=[scatter, scatter_points_whitin_cube3, linje] + cube, layout=layout)
fig = go.Figure(data=[scatter, linje], layout=layout)
# Vise plottet
#fig = go.Figure(data=[scatter, scatter_points_whitin_cube3, scatter_points_whitin_cube4, scatter_points_whitin_cube5, scatter_points_whitin_cube6], layout=layout)#])
fig.show(renderer="browser")




###################### Lager 2D tverrsnittsprofil av aktuelt utsnitt #####################
# Henter ut x- og z-koordinater, ignorerer y
# For å få plot som er parallellt med aksene brukers X3 og Y3
################# Legger til profiltegninger####################
crossection_A = np.array([
                          [-21.1 , 0,  -0],
                          [-15.1 , 0,   4],
                          [-6.2  , 0,   4],
                          [-1.7  , 0,   7],
                          [0     , 0,   7],
                          [4.3   , 0,   7],
                          [10.8  , 0,   2],
                          [12.8  , 0,   2],
                          [17    , 0,   2],
                          [17    , 0,   2]
])

crossection_B = np.array([
                          [-28.45, 0,  -1],
                          [-19.75, 0, 4.8],
                          [-7.8, 0,   4.8],
                          [-3    , 0,   8],
                          [0     , 0,   8],
                          [3     , 0,   8],
                          [10.8  , 0,   2],
                          [14.8  , 0,   2],
                          [20    , 0,  -2],
                          [21.95 , 0,  -2]
])

crossection_C = np.array([
                          [-27.24, 0,  -1],
                          [-19.74, 0,   4],
                          [-9    , 0,   4],
                          [-3    , 0,   8],
                          [0     , 0,   8],
                          [3     , 0,   8],
                          [10.1  , 0,   2],
                          [14.1  , 0,   2],
                          [19.3  , 0,  -2],
                          [21.25 , 0,  -2]
])

crossection_D = np.array([
                          [-32.15, 0,  -1],
                          [-22.4 , 0, 5.5],
                          [-9.75 , 0, 5.5],
                          [-3    , 0,  10],
                          [0     , 0,  10],
                          [3     , 0,  10],
                          [13.4  , 0,   2],
                          [17.4  , 0,   2],
                          [22.6  , 0,  -2],
                          [24.1  , 0,  -2]
])

crossection_E = np.array([
                          [-32.14, 0,  -1],
                          [-20.44, 0, 6.8],
                          [-7.8  , 0, 6.8],
                          [-3    , 0,  10],
                          [0     , 0,  10],
                          [3     , 0,  10],
                          [13.4  , 0,   2],
                          [17.4  , 0,   2],
                          [23.9  , 0,  -3],
                          [26.9  , 0,  -3]
])

crossection_F = np.array([
                          [-32.15, 0,  -1],
                          [-22.4 , 0, 5.5],
                          [-9.75 , 0, 5.5],
                          [-3    , 0,  10],
                          [0     , 0,  10],
                          [3     , 0,  10],
                          [13.4  , 0,   2],
                          [17.4  , 0,   2],
                          [23.9  , 0,  -3],
                          [26.9  , 0,  -3]
])

crossection_G = np.array([
                          [-34.4 , 0,  -1],
                          [-24.65, 0, 5.5],
                          [-9.75 , 0, 5.5],
                          [-3    , 0,  10],
                          [0     , 0,  10],
                          [3     , 0,  10],
                          [13.4  , 0,   2],
                          [17.4  , 0,   2],
                          [23.9  , 0,  -3],
                          [26.9  , 0,  -3]
])

crossection_H = np.array([
                          [-39.25, 0,  -1],
                          [-27.55, 0, 6.8],
                          [-7.8  , 0, 6.8],
                          [-3    , 0,  10],
                          [0     , 0,  10],
                          [3     , 0,  10],
                          [9.8   , 0,   2],
                          [16.7  , 0,   2],
                          [37.05 , 0,  -5.5],
                          [49.5  , 0,  -12]
])

crossection_I = np.array([
                          [-39.2 , 0,  -1],
                          [-27.5 , 0, 6.8],
                          [-7.8  , 0, 6.8],
                          [-3    , 0,  10],
                          [0     , 0,  10],
])

crossection_J = np.array([
                          [-39.85, 0,  -1],
                          [-30.55, 0, 4.8],
                          [-10.8 , 0, 4.8],
                          [-3    , 0,  10],
                          [0     , 0,  10],
])


# Må manuelt velge hvilket profilsnitt som skal brukes
X_c = crossection_C[:, 0] 
Z_c = crossection_C[:, 2]


#################################################
# Lag 2D scatter-plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X3, Z3, color='black', marker='.', s=10, alpha=0.6)
ax.plot(X_c, Z_c, color='red', linestyle='-', marker='.', alpha=0.6)
# Setter titler og akser
ax.set_title("2D Molotverrsnitt")
ax.set_xlabel("X")
ax.set_ylabel("Z")

# Setter akseforhold til 1:1
ax.set_aspect('equal', adjustable='box')

# Vise plottet
plt.grid(True, alpha=0.7)
plt.savefig("2D molotverrsnitt med tilhørende snitttegningsprofil.", dpi=300, bbox_inches='tight')

plt.show()


















############################# Crossection 3D-plot

# # 3D- plot av punkter innnenfor kuben
# x_values = points_within_cube2[:, 0]
# y_values = points_within_cube2[:, 1]
# z_values = points_within_cube2[:, 2]

# # Lag 3D scatter-plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plotter punktene i 3D
# ax.scatter(x_values, y_values, z_values, color='b', marker='o', s=10, alpha=0.6)

# # Setter titler og akser
# ax.set_title("3D Scatter Plot of Points Within Cube")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # Finne minimums- og maksimumsverdier for å sette akseforhold
# max_range = np.array([x_values.max() - x_values.min(),
#                       y_values.max() - y_values.min(),
#                       z_values.max() - z_values.min()]).max() / 2.0

# mid_x = (x_values.max() + x_values.min()) * 0.5
# mid_y = (y_values.max() + y_values.min()) * 0.5
# mid_z = (z_values.max() + z_values.min()) * 0.5

# # Setter grensene slik at alle aksene har samme skala
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)

# # Vise plottet
# plt.show()

##############################
