# Jules Kourelakos Independent Study Spring 2024

## run_experiment.py

Runs experiment and outputs plots detailing random forest regression model's prediction of dependent variables of interest, as well as feature importances.

**Dependencies:** sklearn, pandas, numpy, matplotlib, seaborn

**Usage:** ``python3 run_experiment.py``

## load_csvs.py
Creates single CSV comprised of concatenated data from:
* [UNDP Human Development Index](https://hdr.undp.org/data-center/documentation-and-downloads)
* [Global Data Lab Subnational Human Development Index](https://globaldatalab.org/shdi/)
* [Data Portal for Cities EUI values (from previously procesunicipal validatioon data)](https://dataportalforcities.org/)
* JHU APL Climate Zones

**Dependencies:** geopandas, pandas, rasterio, numpy, tqdm, pyproj 

**Usage:** ``python3 load_csvs.py`` 
