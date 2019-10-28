from rapidboom import EquivArea
from rapidboom import AxieBump

width = 2
# Flight conditions inputs
alt_ft = 50000.

# Setting up for
CASE_DIR = "./"  # axie bump case
PANAIR_EXE = 'panair.exe'
SBOOM_EXE = 'sboom_windows.dat.allow'

# Run
# axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE) # for standard atmo
# untested EquivArea class
equivarea = EquivArea(CASE_DIR, SBOOM_EXE, altitude=alt_ft,
                    deformation='gaussian', weather='standard')#,
                    #altitude=height_to_ground, weather=weather_data)
axiebump = AxieBump(CASE_DIR, PANAIR_EXE, SBOOM_EXE, altitude=alt_ft,
                    deformation='gaussian', weather='standard')
axiebump.MESH_COARSEN_TOL = 0.00035  # 0.000035
axiebump.N_TANGENTIAL = 20
gauss_amp = 1
gauss_loc = 12.5
gauss_std = 1
#loudness = equivarea.run([gauss_amp, gauss_loc, gauss_std])
loudness = axiebump.run([gauss_amp, gauss_loc, gauss_std])
print(loudness)
