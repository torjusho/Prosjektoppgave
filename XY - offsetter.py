#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:23:42 2024

@author: Torjus
"""
#Fjerner fra x og y kolonne



import random
import numpy as np

input_file = "/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/Redusert utgave/Sirevåg  hoved_dsm_20cm_nn2000__.xyz"
output_file = '20cm_20%.xyz'

# Verdiene koordinatene skal transleres med og prosentandel av punkter som skal beholdes
offset_x =  312689.508
offset_y =  6489686.253 
sample_percentage = 25 

# Åpne input-fil og les linje for linje
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if random.uniform(0,100) < sample_percentage:
            
            columns = line.strip().split() # Deler linjen opp i kolonner
        
        # Konverter kolonnene til flyttall (float) for å trekke fra offset-verdier
            x = float(columns[0]) - offset_x
            y = float(columns[1]) - offset_y
            z = float(columns[2])
            r, g, b = columns[3], columns[4], columns[5]
        
        # Skriver de justerte verdiene til output-filen
            outfile.write(f"{x:.3f} {y:.3f} {z} {r} {g} {b}\n")

print(f"Justerte koordinater er lagret i {output_file}")

file_path = '/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/Redusert utgave/20cm_20%.xyz'

data  = np.genfromtxt(file_path)

print(len(data))