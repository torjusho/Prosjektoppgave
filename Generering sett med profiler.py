#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:39:14 2024

@author: Torjus
"""
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

''' Inputs i denne koden er start og sluttpunkt målt fra innerst på moloen til ytterst. Steglengde må settes. 
Nederst i koden må brukt tverrsnitt legges inn manuelt, både for plot, og for hvilken som lagres i filen. 
'''

all_iterations = []
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


''' Itererer gjennom moloen fra definert start/sluttpunkt, og steglengde, for å lage 
    tverrsnittsprofiler 90 grader på senterlinje 
    Skade   1: 
    Skade   2:143 - 157 værst mellom 147-151
    Skade 3.1:210-222 værst mellom 213-217
    Skade 3.2:
    Skade   4: 285, 300, 0.5
    Skade 5.1:
    Skade 5.2:
    Skade   6:
    '''
for length_from_breakwater_head in np.arange(342.5, 372.5, 0.5):
    ############## Punkter for senterlinje hentes inn ###############
    file_path_line = '/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/3Dmodifisert/line_translated_coordinates.xyz'
    center_line  = np.genfromtxt(file_path_line)

    ################# Inputs ###################################
    x_size , y_size, z_size = 100, 0.2, 25 # Geometriske mål av kuben som skal filtrere vekk punkter som ikke er innenfor grenseverdiene som settes.
    center_x, center_y, center_z, rotation_angle = point_and_angle_at_line(center_line, length_from_breakwater_head) # Posisjon og rotasjon

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
    theta = -rotation_angle
    rotation_matrix = np.array([[np.cos(theta),-np.sin(theta),0],
                                [np.sin(theta) ,np.cos(theta) ,0],
                                [0             ,0             ,1]])
         
    #Roterer om origo av kuben
    rotated_matrix = points @ rotation_matrix.T
        
    #Translerer matrisen til det aktuelle punktet på senterlinjen
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
        line = dict(color = 'black', width = 4)
        )
        cube_edges.append(line)

    for edge in edges:
        line = go.Scatter3d(
            x = [points[edge[0],0], points[edge[1],0]],
            y = [points[edge[0],1], points[edge[1],1]],
            z = [points[edge[0],2], points[edge[1],2]],
        mode = 'lines',
        line = dict(color = 'black', width = 4)
        )
        cube_edges2.append(line)
            
    cube  = cube_edges
    cube2 = cube_edges2
    ###############################################
    
    ###################### Definerer kubenses maksimum og minimums verdier #######################
    cube_points_x = shifted_matrix[:, 0]
    cube_points_y = shifted_matrix[:, 1]
    cube_points_z = shifted_matrix[:, 2]

    cube_points_x_2 = points[:, 0]
    cube_points_y_2 = points[:, 1]
    cube_points_z_2 = points[:, 2]

    ################### Data fra punktskyen ##############################
    file_path = '/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/Redusert utgave/20cm_20%.xyz'

    data  = np.genfromtxt(file_path)

    # Hente X, Y, Z og RGB fra datafil 1
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

    ''' Filtrerer her ut punkter som er utenfor grenseverdiene satt av kuben. 
        I første omgang blir det ikke tatt hensyn til rotasjon av kuben, slik at det er max og min x,y og z- verdier som bestemmer
        Andregangsfiltrering skjer etter translering og rotering '''
    ############### Finne maks og minverdier for førstegangsfiltrering av punkter ###########################
    x_min, x_max = np.min(cube_points_x), np.max(cube_points_x)
    y_min, y_max = np.min(cube_points_y), np.max(cube_points_y)
    z_min, z_max = np.min(cube_points_z), np.max(cube_points_z)

    # Definere punkter fra punktskyen som er innenfor maks og min- koordinater av rotert kube
    points_within_cube = data[
        (X >= x_min) & (X <= x_max) &
        (Y >= y_min) & (Y <= y_max) &
        (Z >= z_min) & (Z <= z_max)
    ]

    # Translerer punktene til origo
    points_within_cube[:, 0] -= center_x
    points_within_cube[:, 1] -= center_y
    points_within_cube[:, 2] -= center_z

    # Roterer translerte punkter med invers rotasjonsmatrise
    inverse_rotation_matrix = np.array([[np.cos(-theta),-np.sin(-theta), 0],
                                         [np.sin(-theta),np.cos(-theta) , 0],
                                         [0                      ,0                       , 1]])

    rotated_points_within_initial_cube = points_within_cube[:, :3] @ inverse_rotation_matrix.T

    # Punktene er nå rotert med motsatt verdi for å filtrere punktene innenfor en kube som står vinkelrett på aksesystemet
    # Kuben er nå vinkelrett på aksesystemet, og punktene er rotert, kubens hjørnekoordinater blir da grenseverdier for andregangsfiltrering
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

    X3 = points_within_cube2[:, 0]
    Y3 = points_within_cube2[:, 1]
    Z3 = points_within_cube2[:, 2]




    ''' Lager 2D tverrsnittsplott av aktuell profil
    Henter ut x- og z-koordinater, ignorerer y
    For å få plot som er parallellt med aksene brukers X3 og Y3 i plottet
    Legger til tverrsnittstegninger '''
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


    ''' Må manuelt velge hvilket profilsnitt som skal brukes '''
    X_c = crossection_F[:, 0]
    Z_c = crossection_F[:, 2]


    #################################################
    # Lag 2D scatter-plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X3, Z3, color='black', marker='.', s=10, alpha=0.6)
    ax.plot(X_c, Z_c, color='red', linestyle='-', marker='.', alpha=0.6)
    # Setter titler og akser
    ax.set_title(f"2D Molotverrsnitt for snitt {length_from_breakwater_head} m fra startpunkt")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # Setter akseforhold til 1:1
    ax.set_aspect('equal', adjustable='box')

    # Vise plottet
    plt.grid(True)
    plt.show()
    
    ''' Koordinater for punkter i alle profiler blir lagret i xyz-fil, og crossection blir lagt inn først '''
    
    if not all_iterations:
        all_iterations.append(crossection_F.tolist())
        
    all_iterations.append(points_within_cube2.tolist())

# Lagre til fil i ønsket format
with open("profiler_fra_crossection.xyz", "w") as f:
    f.write(str(all_iterations))

#print(all_iterations) # Dette er det som lagres i profiler_fra_crossection.xyz