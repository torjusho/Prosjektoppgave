#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:13:39 2024

@author: Torjus
"""
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d

############################### Inputs ###############################

W_50 = 13300 # middelvekt på stein i blokklag i kg
rho = 2700   # tetthet stein
Dn50 = 1.7
#(6 * W_50 / (rho * np.pi))**(1/3)  # Lengde på kubiske steinblokk i m (Nominell diameter for steinblokk)


interp_points_on_x_axis = 500 # antall interpolerte punkter

# Helning fra skulder til vannkant
theta = np.arctan(1/1.5) 

z_for_avg_rec = 0.3   # Høyde over sjøkartnull for å beregne recession. For å fjerne tang og vann fra beregningene

# upper_z_limit_rec_manual_mode = 5    # Koden skal egentlig finne ut av dette selv, men noen ganger gjør den ikke det, og manuell input kreves
#######################################################################
#Justeringer 
#mark_profile_nr = 13

step_length = 0.2

filtervalue  = 2



# Hente filen som inneholder as_built_profil, og tverrsnittsprofiler
with open('/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/3Dmodifisert/Samples20cm100%/Skade 2/0.2/147-151-0.2.xyz', 'r') as f:
    file_content = f.read()


# with open("/Users/Torjus/Documents/Bygg og miljøteknikk/9. semester/Prosjektoppgave/3Dmodifisert/profiler_fra_crossection.xyz", "r") as f:
#     file_content = f.read()

# Konverter tekstinnholdet til en Python-liste
data = ast.literal_eval(file_content)

# Første element i listen representerer as_built_crossection-punktene
as_built_crossection = np.array(data[0])

middle_of_berm = (as_built_crossection[1,0]+as_built_crossection[2,0])/2

length_of_slope_from_berm_to_water = np.sqrt((as_built_crossection[0,0]-as_built_crossection[1,0])**2+(as_built_crossection[0,2]-as_built_crossection[1,2])**2)

# print('Profil nr', mark_profile_nr, 'av', len(data)-1, 'profiler er markert i plot')
# Figuregenskaper
plt.rc('font', family='Arial', size=12)
plt.rc('figure', facecolor='w')
plt.rc('lines', linewidth=1.5)

# Plotter alle profilene i samme figur
plt.figure(figsize=(10, 6))

# Plotter som bygget tverrsnitt(as_built_crossection)
plt.plot(as_built_crossection[:, 0], as_built_crossection[:, 2], label='Snittegning', color='blue', linestyle='-', linewidth=0.7)

# Liste for å lagre Sd-verdier
Sd_values = []
de_values = []
Rec_values = []


# Rotasjonsmatrise
rotation_matrix = np.array([[np.cos(theta) , 0, np.sin(theta)],
                            [0             , 1,             0],
                            [-np.sin(theta), 0, np.cos(theta)]])

# Iterer over alle profilene tatt fra punkskyen
for idx in range(1, len(data)):
    profile = np.array(data[idx])
    # Sorter profilene etter x- verdier
    profile = profile[np.argsort(profile[:, 0])]
    
    
    # Setter grenser for x < -15 and z > 0 for Sd - beregninger
    profile_filtered = profile[(profile[:, 0] < middle_of_berm) & (profile[:, 2] > -1)]
    
    # Plotter profiler fra punksky
   # plt.plot(profile[:, 0], profile[:, 2], color='red', linestyle='-', linewidth=0.2, label='Tverrsnittsprofiler' if idx == 1 else "")
    

    # Hvis en av profilene skal markeres i plottet kan denne brukes, a = profilnummer som skal markeres
    a = 20
    if idx == a:
        plt.plot(profile[:, 0], profile[:, 2], label=f'Profil {idx}', color='red', linestyle='-', linewidth=1)
#    if idx == 2:
#        plt.plot(profile[:, 0], profile[:, 2], label=f'Profil {idx}', color='green', linestyle='-', linewidth=1)
# #    else:
#        plt.plot(profile[:, 0], profile[:, 2], label=f'After reshaping - Profile {idx}', color='black', linestyle='-', linewidth=0.2)


    # Interpolerer as_built_crossection for å matche distanse mellom punktene fra punkskyen
    interp_func = interp1d(as_built_crossection[:, 0], as_built_crossection[:, 2], kind='linear', fill_value='extrapolate')
    profile_interp = interp_func(profile_filtered[:, 0])

    # Regner ut forskjellen i z-retning mellom interpolert som-bygget profil, og faktiske profiler
    dZ = profile_interp - profile_filtered[:, 2]
    dZ[dZ < 0] = 0  # Kun ersosjon ( bare negative verdier)  

    # Beregner dx som gjennomsnittet av forskjellene i x- koordinater Calculate dx as mean of differences in x-coordinates
    dx = np.mean(np.diff(profile_filtered[:, 0]))

    # Beregner arealet av erodert område, og antall steiner som er flyttet på
    Aerode = dx * trapz(dZ)  # Areal i m^2
    Sd = round(Aerode / Dn50**2, 3)  # Antall steinblokk som er flyttet på 

    # Legg til Sd verdier i listen
    Sd_values.append(Sd)
    # if idx==a:
    #     plt.fill_between(profile_filtered[:, 0], profile_filtered[:, 2], profile_interp, where=(profile_interp > profile_filtered[:, 2]), color='orange', alpha=0.5, label='Erodert areal ($A_e$) [$m^2$]')

    
    ################ de-beregninger #################
    # Flytter punkter til origo, slik at 0m alt på as_built_profile's venstre side ville ligget i origo
    profile_shifted = profile.copy()
    profile_shifted[:, 0] = profile[:, 0] - as_built_crossection[0, 0]  # Juster x-verdier
    profile_shifted[:, 2] = profile[:, 2] - as_built_crossection[0, 2]  # Juster z-verdier
    # Roterer slik at linjen mellom skulderen og vannkanten likker langs med x-aksen
    rotated_profile = profile_shifted @ rotation_matrix.T
#    plt.plot(rotated_profile[:, 0], rotated_profile[:, 2], color='blue', linestyle='-', linewidth=0.2) # Visualiserer at det er riktig

    # Finn de første punktene med z- koordinater mindre enn null etter rotasjonen
    min_de_value = None
    for i in range(len(rotated_profile)):
        if rotated_profile[i, 0] < length_of_slope_from_berm_to_water and rotated_profile[i, 2] < 0:
            if min_de_value is None or rotated_profile[i, 2] < min_de_value:
                min_de_value = round(rotated_profile[i, 2], 3)

    # Legg til den minste d_e-verdien fra denne profilen, hvis den finnes (Negativ verdi slik at minste betyr dypeste dybde på erodert område)
    if min_de_value is not None:
        de_values.append(- min_de_value)

    ######################## Recession beregninger ########################
    
    profile_shifted_2 = profile.copy()
    profile_shifted_2[:, 0] = profile[:, 0] - as_built_crossection[1, 0]  # Juster x-verdier
    profile_shifted_2[:, 2] = profile[:, 2] - as_built_crossection[1, 2]  # Juster z-verdier
    
    rec_value = None
    for i in range(len(profile_shifted_2)):
        if profile_shifted_2[i, 0] > 0 and profile_shifted_2[i, 2] > -0.2:   # Setter grense til -0.2 for å ikke misse alt
            if rec_value is None or profile_shifted_2[i, 0] < rec_value:
                rec_value = round(profile_shifted_2[i, 0], 3)

    # Legg til den største rec-verdien fra denne profilen, hvis den finnes
    if rec_value is not None:
        Rec_values.append(rec_value)
    
#     plt.plot(profile_shifted_2[:, 0], profile_shifted_2[:, 2], label=f'After reshaping - Profile {idx}', color='red', linestyle='-', linewidth=0.2)
    


# Sett like skala på x- og y-aksen
plt.gca().set_aspect('equal', adjustable='box')
# Label the axes
plt.xlabel('x [m]')
plt.ylabel('z [m]')
#plt.title('As built and after reshaping profiles')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
plt.legend(loc='upper right')
plt.savefig("As built and after reshaping profiles at crossection D.png", dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
print(f'Antall profiler = {len(data)-1}')
print('Midt på skulder =', middle_of_berm)
print('Lengden fra vannkant til skulder =', length_of_slope_from_berm_to_water)
print()
print("Rec values for each profile after reshaping:")
print(Rec_values)
print()
max_index_rec = Rec_values.index(max(Rec_values))
print(f'Max Rec value: {max(Rec_values)} at profile {max_index_rec + 1}')
print()
# Dele hvert element i listen på Dn50
dimless_rec = [value / Dn50 for value in Rec_values]
dimless_rec = [round(value, 3) for value in dimless_rec]
print("Dimensionless Rec/Dn50 values for each profile after reshaping:")
print(dimless_rec)
print()
# Print Sd values
print("Sd values for each profile after reshaping:")
print(Sd_values)
print()
print('Sd mean =', np.mean(Sd_values))
print()
max_index = Sd_values.index(max(Sd_values))
print(f'Max Sd value: {max(Sd_values)} at profile {max_index + 1}')
print()
print("d_e values for each profile after reshaping:")
print(de_values)
print()
max_index2 = de_values.index(max(de_values))
print(f'Max d_e value: {max(de_values)} at profile {max_index2 + 1}')
print()



Rec_mean = round(np.mean(Rec_values),3)
print(f'Mean rec value = {Rec_mean}')
print()
dimless_Rec_mean = Rec_mean/Dn50
print(f'Mean rec/Dn50 value = {dimless_Rec_mean}')









# Beregner en felles x-akseområde for alle profilene
common_x = np.linspace(min(as_built_crossection[:, 0]), max(as_built_crossection[:, 0]), interp_points_on_x_axis)

# Gjennomsnittlig profil for eroderte profiler 
'''Calculate the average profile after the storm using interpolation''' 
avg_profile = np.zeros((len(common_x), 3))  # Initiser avg_profile til å inneholde x, y og z- verdier
count = 0

for idx in range(1, len(data)):
    profile = np.array(data[idx])
    # Sorter etter x-verdier
    profile = profile[np.argsort(profile[:, 0])]

    # Interpoler profilen til dne felles x-aksen 
    interp_func_z = interp1d(profile[:, 0], profile[:, 2], kind='linear', fill_value='extrapolate')
    interpolated_z = interp_func_z(common_x)

    # Setter y-verdier til 0 fordi det er 2D-plott der punkter for hver steglengde samles
    interpolated_y = np.zeros_like(common_x)

    # Lager interpolerte profiler med x, y, og z- verdier
    interpolated_profile = np.vstack((common_x, interpolated_y, interpolated_z)).T

    # Legger til den gjennomsnittlige profilen 
    avg_profile += interpolated_profile
    count += 1

# Regner ut den gjennomsnittlige profilen
avg_profile /= count

# Filtrerer den gjennomsnittlige profilen basert på den felles x-akse < middle_of_berm og avg_profile[:, 2] > -1
filtered_indices = (common_x < middle_of_berm) & (avg_profile[:, 2] > -1)
common_x_filtered = common_x[filtered_indices]
avg_profile_filtered = avg_profile[filtered_indices]

# Interpolerer as_built_crossection til å matcha den felles x-aksen
interpolated_as_built_profile_func = interp1d(as_built_crossection[:, 0], as_built_crossection[:, 2], kind='linear', fill_value='extrapolate')
as_built_interpolated = interpolated_as_built_profile_func(common_x_filtered)

# Beregner høydeforskjellen mellom den interpolerte som-bygget profilen og profilene fra punkskyen
dZ = as_built_interpolated - avg_profile_filtered[:, 2]
dZ[dZ < 0] = 0  # Kun erosjon (bare negative verdier)

# Beregner dx som gjennomsnittlig av forskjell i x-koordinater Calculate dx as mean of differences in x-coordinates
dx = np.mean(np.diff(common_x_filtered))

# Beregner erodert areal av gjennomsnittsprofil og antall steiner som har flyttet på seg
Aerode = dx * trapz(dZ)  # Areal i m^2

Sd = round(Aerode / Dn50**2, 3)  # Antall stein som har flyttet på seg

Sd_mean_value = Sd
print()
print(f'Sd value for the mean profile: {Sd_mean_value}')


#print(as_built_crossection)
#print('Average profile',avg_profile)

################################### Gjennomsnittlig de - beregninger ################################
# Flytt punkter til origo, slik at 0m alt på as_built_profiles venstre side ville ligget i origo
avg_profile_shifted = avg_profile.copy()
avg_profile_shifted[:, 0] = avg_profile[:, 0] - as_built_crossection[0, 0]  # Juster x-verdier
avg_profile_shifted[:, 2] = avg_profile[:, 2] - as_built_crossection[0, 2]  # Juster z-verdier
avg_rotated_profile = avg_profile_shifted @ rotation_matrix.T
#    plt.plot(rotated_profile[:, 0], rotated_profile[:, 2], color='blue', linestyle='-', linewidth=0.2)

    # Finn de første punktene med x-koordinater mindre enn null etter rotasjonen
min_de_value = None
for i in range(len(avg_rotated_profile)):
    if avg_rotated_profile[i, 0] < length_of_slope_from_berm_to_water and avg_rotated_profile[i, 2] < 0:
        if min_de_value is None or avg_rotated_profile[i, 2] < min_de_value:
            min_de_value = round(avg_rotated_profile[i, 2], 3)

    # Legg til den minste d_e-verdien fra denne profilen, hvis den finnes
if min_de_value is not None:
    avg_de_value = (- min_de_value)
print()
print('gjennomsnittlig d_e verdi =', avg_de_value)
######################################################################################
plt.figure(figsize=(10, 6))

# Plotter som bygget tverrsnitt(as_built_crossection)
plt.plot(as_built_crossection[:, 0], as_built_crossection[:, 2], label='Snittegning', color='blue', linestyle='-', linewidth=0.5)

for idx in range(1, len(data)):
    profile = np.array(data[idx])
    # Sorter profilene etter x- verdier
    profile = profile[np.argsort(profile[:, 0])]
    
    
    # Setter grenser for x < -15 and z > 0 for Sd - beregninger
    profile_filtered = profile[(profile[:, 0] < middle_of_berm) & (profile[:, 2] > -1)]
    
    # Plotter profiler fra punksky
    plt.plot(profile[:, 0], profile[:, 2], label=None, color='red', linestyle='-', linewidth=0.2, alpha=0.7)

    # Hvis en av profilene skal markeres i plottet kan denne brukes
    if idx == max_index:
        plt.plot(profile[:, 0], profile[:, 2], label=f'Maks $S_d$ ved profil {idx + 1}', color='green', linestyle='-', linewidth=1)
    
    if idx == max_index2:
        plt.plot(profile[:, 0], profile[:, 2], label=f'Maks $d_e$ ved profil {idx + 1}', color='black', linestyle='-', linewidth=1)
   
    if idx == max_index_rec:
        plt.plot(profile[:, 0], profile[:, 2], label=f'Maks $Rec$ ved profil {idx + 1}', color='red', linestyle='-', linewidth=1)

# Sett like skala på x- og y-aksen
plt.gca().set_aspect('equal', adjustable='box')
# Label the axes
plt.xlabel('x [m]')
plt.ylabel('z [m]')
#plt.title('As built and after reshaping profiles')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
plt.legend(loc='upper right')
plt.savefig("Utplukkede profiler.png", dpi=300, bbox_inches='tight')
plt.show()


############################### Plot av gjennomsnittsprofil ##########################
plt.figure(figsize=(10, 6))

# Plotter som-bygget-profilen
plt.plot(as_built_crossection[:, 0], as_built_crossection[:, 2], label='Snittegning', color='blue', linestyle='-')
plt.fill_between(common_x_filtered, avg_profile_filtered[:, 2], as_built_interpolated, where=(as_built_interpolated > avg_profile_filtered[:, 2]), color='orange', alpha=0.5, label='Erodert areal ($A_e$)[$m^2$]')

# Plotter gjennomsnittsprofilen
plt.plot(avg_profile[:, 0], avg_profile[:, 2], label='Gjennomsnittsprofil', color='red', linestyle='-', linewidth=1)
plt.xlabel('x [m]')
plt.ylabel('z [m]')
#plt.title('Comparison between mean reshaped profiles and as-built profile ')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='upper right')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Sett ticks for å få 1x1 grid spacing
#plt.xticks(np.arange(int(min(as_built_crossection[:, 0])), int(max(as_built_crossection[:, 0])) + 2, 1))
#plt.yticks(np.arange(int(min(avg_profile[:, 2])), int(max(avg_profile[:, 2])) + 1, 1))
plt.savefig("Comparison between mean reshaped profiles and as-built profile", dpi=300, bbox_inches='tight')
plt.show()

#######################################################################################

############# Gjennomsnittlig recession for gjennomsnittlig profil ################# Gjennomsnittlig recession langs hele eroderte område av gjennomsnittlig profil
# # Lagrer som nye variabler for å ikke klusse til originale
# as_built_crossection_rec = as_built_crossection
# avg_profile_rec = avg_profile

# # Steg 1: Interpoler as_built_tverrsnitt (kun mellom de første 6 punktene)
# as_built_crossection_to_interpolate = as_built_crossection_rec[:5]   # Må spesifisere at kun de første 6 punktene skal være med i interpoleringen for å ikke få konflikt med skulder på innsiden av moloen
# interp_function_rec = interp1d(as_built_crossection_to_interpolate[:, 2], as_built_crossection_to_interpolate[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
# as_built_interpolated_x_rec = interp_function_rec(avg_profile_rec[:, 2])
# as_built_interpolated_rec = np.column_stack((as_built_interpolated_x_rec, np.zeros_like(avg_profile_rec[:, 2]), avg_profile_rec[:, 2]))

# # Steg 2: Sett grenser og filtrer for begge profilene
# # FInn den øvre z- grensen som verdien før første verdi der to z- verdier etter hverandre er like ( indikerer at vi er på skuldra)
# upper_z_limit_rec = None
# for i in range(1, len(as_built_interpolated_rec[:, 2])):
#     if as_built_interpolated_rec[i, 2] == as_built_interpolated_rec[i - 1, 2]:
#         upper_z_limit_rec = as_built_interpolated_rec[i - 2, 2]
#         break
# if upper_z_limit_rec is None:
#     upper_z_limit_rec = upper_z_limit_rec_manual_mode #'Standardverdi hvis den ikke finner noen verider som er like' # Skal gi error
# if upper_z_limit_rec == upper_z_limit_rec_manual_mode:
#     print('Manuell inputverdi for upper_z_limit_rec er brukt, sjekk om riktig verdi er lagt inn i upper_z_limit_manual_mode under inputs, øverst i koden')
# # Filtrer for negative x verdier og verdier som er innenfor z- grensene som settes 
# avg_profile_filtered_rec = avg_profile_rec[(avg_profile_rec[:, 0] < 0) & (avg_profile_rec[:, 2] >= 0.3) & (avg_profile_rec[:, 2] <= upper_z_limit_rec)]
# as_built_filtered_rec = as_built_interpolated_rec[(as_built_interpolated_rec[:, 0] < 0) & (as_built_interpolated_rec[:, 2] >= 0.3) & (as_built_interpolated_rec[:, 2] <= upper_z_limit_rec)]

# # Step 3: Match interpolert som-byggetprofil med gjennomsnittsprofil
# # Definer dz-intervall
# dz_rec = 0.1

# # Beregn avstander i x- retning
# z_values_rec = np.arange(avg_profile_filtered_rec[:, 2].min(), avg_profile_filtered_rec[:, 2].max() + dz_rec, dz_rec)
# distances_rec = []

# for z_rec in z_values_rec:
#     avg_profile_points_rec = avg_profile_filtered_rec[np.isclose(avg_profile_filtered_rec[:, 2], z_rec, atol=dz_rec / 2)]
#     as_built_points_rec = as_built_filtered_rec[np.isclose(as_built_filtered_rec[:, 2], z_rec, atol=dz_rec / 2)]
    
#     if len(avg_profile_points_rec) > 0 and len(as_built_points_rec) > 0:
#         distance_rec = abs(avg_profile_points_rec[0, 0] - as_built_points_rec[0, 0])
#         distances_rec.append((z_rec, distance_rec))


# for z_rec, distance_rec in distances_rec:
#     print(f"z: {z_rec:.2f}, Rec-distance: {distance_rec:.4f}")

# # Plot gjennomsnittsprofil og som-bygget
# plt.figure(figsize=(10, 6))
# plt.plot(as_built_crossection[:, 0], as_built_crossection[:, 2], label='Snittegning', color='blue')
# plt.plot(avg_profile_rec[:, 0], avg_profile_rec[:, 2], label='Gjennomsnittsprofil', color='red')

# # Plot linjer der avstanden mellom dem i x- retning
# for z_rec, distance_rec in distances_rec:
#     avg_x = avg_profile_filtered_rec[np.isclose(avg_profile_filtered_rec[:, 2], z_rec, atol=dz_rec / 2)][0, 0]
#     as_built_x = as_built_filtered_rec[np.isclose(as_built_filtered_rec[:, 2], z_rec, atol=dz_rec / 2)][0, 0]
#     plt.plot([avg_x, as_built_x], [z_rec, z_rec], color='g', linestyle=':', linewidth=0.5)

# plt.xlabel('x [m]')
# plt.ylabel('z [m]')
# #plt.title('Average rec for average profile')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
# plt.savefig("Average rec for average profile", dpi=300, bbox_inches='tight')

# plt.show()

# average_distance = np.mean([distance for _, distance in distances_rec])
# print(f"Average reccession: {average_distance:.3f}")

###########################################################################

# Plot ale profilene i den samme figuren med riktig steglengde m y-offset in 3D
fig = plt.figure(figsize=(16, 20))
ax = fig.add_subplot(111, projection='3d')

# Definer fargekart for grønne linjer
colors = plt.cm.viridis(np.linspace(0, 1, len(data) - 1))

# Iterer over alle profilene
for idx in reversed(range(1, len(data))):
    profile = np.array(data[idx])
    # Sorter etter x-verdier
    profile = profile[np.argsort(profile[:, 0])]

    # Filtrer kun negative x-verdier
    profile = profile[profile[:, 0] < middle_of_berm + -middle_of_berm/2 + filtervalue]

    # Sett alle y-verdiene lik nummeret på iterasjonen (idx - 1 for å starte på 0)
    y_offset = (idx - 1) * step_length

    # Juster y-verdiene for profilen etter stormen
    profile[:, 1] = y_offset

    # Plot profiler (grønn linje med unik farge per iterasjon)
    ax.plot(profile[:, 0], profile[:, 1], profile[:, 2], 
            label=f'After Reshaping - Profile {idx} (offset by {y_offset} m)', 
            color=colors[idx - 1], linestyle='-', linewidth=3.5)

    # Plot profiler med samme y-offset som profilen etter stormen (blå linje)
    as_built_profile = as_built_crossection[as_built_crossection[:, 0] < middle_of_berm + -middle_of_berm/2+ filtervalue] 
    ax.plot(as_built_profile[:, 0], 
            np.full_like(as_built_profile[:, 0], y_offset), 
            as_built_profile[:, 2], 
            color='black', linestyle='-', linewidth=0.7, 
            label=f'As built profile - Profile {idx} (offset by {y_offset} m)')

ax.set_xlabel('x [m]            ', fontsize=20)
ax.set_ylabel('  y [m]', fontsize=20)
ax.set_zlabel('z [m]', fontsize=20)
#ax.set_title('Profiles with negative X values compared to as-built model')
ax.tick_params(axis='both', which='major', labelsize=17)  # Størrelse for store ticks
#ax.tick_params(axis='both', which='minor', labelsize=10)  # Størrelse for små ticks
ax.tick_params(axis='z', which='minor', labelsize=17)  # Størrelse for små ticks


# Lik skalering på aksene
def set_aspect_equal_3d(ax):
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_size = max(extents[:, 1] - extents[:, 0]) / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - max_size, ctr + max_size)

set_aspect_equal_3d(ax)

# Vinker og zoom for visning av plot
ax.view_init(elev=35, azim=-135)
x_center = np.mean(ax.get_xlim3d())
y_center = np.mean(ax.get_ylim3d())
z_center = np.mean(ax.get_zlim3d())

zoom = 0.3
ax.set_xlim3d([x_center - zoom * (ax.get_xlim3d()[1] - ax.get_xlim3d()[0]), x_center + zoom * (ax.get_xlim3d()[1] - ax.get_xlim3d()[0])])
ax.set_ylim3d([y_center - zoom * (ax.get_ylim3d()[1] - ax.get_ylim3d()[0]), y_center + zoom * (ax.get_ylim3d()[1] - ax.get_ylim3d()[0])])
ax.set_zlim3d([z_center - zoom * (ax.get_zlim3d()[1] - ax.get_zlim3d()[0]), z_center + zoom * (ax.get_zlim3d()[1] - ax.get_zlim3d()[0])])


# Lagre og vis plot
plt.savefig("Profiles compared to as-built model at seaward-side", dpi=300, bbox_inches='tight')
plt.show()


# Plot høydeforskjellen mellom som-bygget profil og faktiske profiler i 3D med 1 m mellom
fig = plt.figure(figsize=(8, 16))
ax = fig.add_subplot(111, projection='3d')

# Iterer over alle profiler
for idx in reversed(range(1, len(data))):
    profile = np.array(data[idx])
    # Sort profile by x-value
    profile = profile[np.argsort(profile[:, 0])]

    # Interpoler som-byggetprofil slik at den matcher avstander mellom punkter i de faktiske profilerne
    interp_func = interp1d(as_built_crossection[:, 0], as_built_crossection[:, 2], kind='linear', fill_value='extrapolate')
    profile_interp = interp_func(profile[:, 0])

    # Beregn forskjellen i høyde mellom interpolert som-bygget profil og faktiske profiler
    dZ = profile[:, 2] - profile_interp

    # Sett negative forskjeller til 0
    dZ[dZ > 0] = 0

    # Tilpass offsett for hver profil i y-retning
    y_offset = (idx - 1) * step_length

    # Plot forskjellen i høyde (dZ) mot x-retning med definert mellomrom mellom profiler
    ax.plot(profile[:, 0], np.full_like(profile[:, 0], y_offset), - dZ, color=colors[idx - 1], linewidth=1.8, label=f'Profile {idx}')

ax.set_xlabel('x [m]', fontsize=20)
ax.set_ylabel('y [m]', fontsize=20)
ax.set_zlabel('$\Delta$z [m]', fontsize=20)
#ax.set_title('Difference in elevation between as-built drawings and drone-scan profiles')

ax.tick_params(axis='both', which='major', labelsize=20)  # Størrelse for store ticks
#ax.tick_params(axis='both', which='minor', labelsize=10)  # Størrelse for små ticks
ax.tick_params(axis='z', which='minor', labelsize=20)  # Størrelse for små ticks




plt.subplots_adjust(left=0.1, right=10, top=0.9, bottom=0.1)


plt.savefig("Difference in elevation between as-built drawings and drone-scan profiles", dpi=300, bbox_inches='tight')

# Vis 3D plot
plt.show()


# # Plotter den midterste profilen
# mid_idx = len(data) // 2
# if len(data) % 2 == 0:
#     mid_idx -= 1  # Velg en av de to midterste profilene hvis det er et oddetall av profiler

# mid_profile = np.array(data[mid_idx])
# mid_profile = mid_profile[np.argsort(mid_profile[:, 0])]

# plt.figure(figsize=(10, 6))
# plt.plot(mid_profile[:, 0], mid_profile[:, 2], label=f'Tverrsnittsprofil', color='red', linestyle='-', linewidth=0.5)
# plt.plot(as_built_crossection[:, 0], as_built_crossection[:, 2], label='Snittegning', color='blue', linestyle='-', linewidth=1.5)

# plt.xlabel('x [m]')
# plt.ylabel('z [m]')
# #plt.title('Profile in the middle of sample, compared to as built profile')
# plt.legend(loc='upper right')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
# plt.savefig("Profile in the middle of sample, compared to as built profile", dpi=300, bbox_inches='tight')

# plt.show()
