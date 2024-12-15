XY-translering og reduksjon av punktsky:
- Brukes for å redusere punktskyen til en passende størrelse for enklere håndterbarhet under utvikling av koden.
- Brukes for å translere punktene til origo i koordinatsystemet for enklere behandling og visuell kontroll.

Koordinattransformer og generering av senterlinje:
- Tar inn en liste med punkter i NGO 1948 akse 1 som representerer senterlinjen i moloen. Legger inn radier for kurver langs senterlinjen for interpolering av kurvepunkter
- Tranformerer deretter senterlinjepunkter fra NGO  1948 akse 1 til UTM sone 32 og translerer mes samme verdier som i "XY-translering og reduksjon av punktsky" slik at de kan nøyaktig overlappes med punkter i punkskyen

3DPunktskyplot og crossection:
- Ble brukt under utvikling av koden, og senere for visualisering
- Tar inn punktskydata og senterlinje og plotter sammen
- Lager prisme som definerer grenseverdier for uttak av punkter til tverrsnittsplott og plotter visualisering i 3D

Generering sett med profiler:
- I praksis samme kode som "3DPunktskyplot og crossection", men tar inn start og sluttverdi for lengde langs senterlinjen, i tilleg til steglengde og iterer gjennom og lager tverrsnittsplot som lagres i en datafil sammen med tverrsnittstegninger av moloen

Final_plots:
- Visualisering av profiler
- Genererer enkeltprofiler, gjennomsnittsprofiler, profiler ved siden av hverandre og forskjell i deltaZ
- Returnerer maksverdier for Rec, S_d og d_e for hver profil og gjennomsnitt for profilsettet
