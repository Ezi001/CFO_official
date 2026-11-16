import h5py



H5_PATH = "data/nordic/nordic_seas.h5"

with h5py.File(H5_PATH, "r") as f:

    print(list(f.keys()))  # prints all groups in the h5 file

    log_bath = f["log_bathymetry/data"][()]   # [1, 128, 128, 1]

    land_mask = f["land_mask/data"][()]        # [1, 128, 128, 1]

    traject = "0001"
    x   = f[f"{traject}/data"][()]    # [112, 128, 128, 1]
    t   = f[f"{traject}/grid/t"][:]    # [112,]
    lat = f[f"{traject}/grid/lat"][()]         # [128,128]
    lon = f[f"{traject}/grid/lon"][()]         # [128,128]

 