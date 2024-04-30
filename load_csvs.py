import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import numpy as np
from tqdm import tqdm
from pyproj import Proj, transform


national_hdi_filepath = 'human_development_index/HDR23-24_Composite_indices_complete_time_series.csv'
subnational_hdi_codes_filepath = 'GDL-Human-Development-(2021)-subnational-codes.csv'
subnational_hdi_data_filepath = 'GDL-Human-Development-(2021)-data.csv'
climate_zones_geotiff_filepath = 'JHU_climate_zones/climate_zones.tif'
DPFC_cities_with_EUI_values_filepath = 'DPFC_cities_EUI_values.csv'


#Takes JHU APL climate zones GeoTiff and a lat/long point; 
#returns climate zone point falls within
def get_closest_climate_zone(geotiff_file, latitude, longitude):
    with rasterio.open(geotiff_file) as src:
        row, col = src.index(longitude, latitude)

        # Read the value of the pixel closest to the input coordinates
        value = src.read(1, window=((row, row+1), (col, col+1)))

    return value[0][0]


def concat_DPFC_and_climate_zones(input_csv_file, geotiff_file):
    df = pd.read_csv(input_csv_file, encoding='latin-1')
    df = df.dropna(subset=["lat", "lng"])

    climate_zones = []
    for index, row in df.iterrows():
        if pd.isna(row['lat']) or pd.isna(row['lng']): continue
        climate_zone = get_closest_climate_zone(geotiff_file, row['lat'], row['lng'])
        climate_zones.append(climate_zone)

    df['Climate_Zone_JHU'] = climate_zones

    return df


#Takes shapefile from Global Data Lab of the codes they use to refer to subnational regions 
#and a dataframe of of lat/long points;
#returns dataframe with correct Global Data Lab subnational regional code for each lat/long point
def get_gdlcode(shapefile_path, lat_long_df):
    gdf = gpd.read_file(shapefile_path)
    geometry = [Point(lon, lat) for lat, lon in zip(lat_long_df['latitude'], lat_long_df['longitude'])]
    gdf_points = gpd.GeoDataFrame(geometry=geometry, crs=gdf.crs)
    joined = gpd.sjoin(gdf_points, gdf, how="left", op="within")
    gdlcode_values = joined['gdlcode']
    result_df = pd.DataFrame({
        'Latitude': [point.y for point in joined.geometry],
        'Longitude': [point.x for point in joined.geometry],
        'gdlcode': gdlcode_values
    })

    return result_df


def read_csvs():
    hdi = pd.read_csv(national_hdi_filepath, encoding='latin-1')
    dpfc_and_climate = concat_DPFC_and_climate_zones(DPFC_cities_with_EUI_values_filepath, climate_zones_geotiff_filepath)
    hdi_dpfc_climate = pd.merge(dpfc_and_climate, hdi, on='country')
    subnational_hdi_codes = pd.read_csv(subnational_hdi_codes_filepath, encoding='latin-1')
    hdi_dpfc_climate_with_subnational_hdi_codes = pd.merge(hdi_dpfc_climate, subnational_hdi_codes, left_on=['latitude', 'longitude'], right_on=['Latitude', 'Longitude'], how='inner')
    subnational_hdi_data = pd.read_csv(subnational_hdi_data_filepath)
    result = pd.merge(hdi_dpfc_climate_with_subnational_hdi_codes, subnational_hdi_data, left_on='gdlcode', right_on='GDLCODE', how='inner')
    result.drop('GDLCODE', axis=1, inplace=True)
    

if __name__ == "__main__":
    result = read_csvs()
    result.to_csv('final_subnational.csv')