import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rioxarray.merge import merge_datasets
from osgeo import gdal
from xarray.ufuncs import logical_or  as xr_or
import yaml
from datetime import datetime
import rasterio 
import logging
import os
import glob


def unpack_bits(land_cover_endcoding, data_array, cover_type):
    """
	Description:
		Unpack bits for end of ls7 and ls8 functions 
	-----
	Input:
		land_cover_encoding(dict hash table) land cover endcoding provided by ls7 or ls8
        data_array( xarray DataArray)
        cover_type(String) type of cover
	Output:
        unpacked DataArray
	"""
    boolean_mask = np.isin(data_array.values, land_cover_endcoding[cover_type])
    return xr.DataArray(boolean_mask.astype(bool),
                        coords=data_array.coords,
                        dims=data_array.dims,
                        name=cover_type + "_mask",
                        attrs=data_array.attrs)

def ls_unpack_qa(data_array, cover_type):
    """
    Function originally from dc_mosaic.py

    For l4, l5, l7, and l8, the pixel_qa band is a bit band where each bit 
    corresponds to some surface condition. The 6th bit corresponds to clear, 
    and the 7th bit corresponds to water. Note that a pixel will only be flagged
    as water if it is also not cloud, cloud shadow, etc.
    """
    boolean_mask = np.zeros(data_array.shape, dtype=bool)
    data_array = data_array.astype(np.int64)

    if cover_type == 'clear':
        # 6th bit is clear
        boolean_mask |=  ((data_array & 0b0000000001000000) != 0)
    elif cover_type == 'water':
        # 7th bit is water
        boolean_mask |= ((data_array & 0b0000000010000000) != 0)
    else:
        raise ValueError(f"Cover type {cover_type} not supported for Landsat 8 yet")
    return boolean_mask


def sen2_unpack_qa(data_array, cover_type):
    """
    Function originally from dc_mosaic.py 

    Changed values to correspond to the Sentinel-2 Scene Classification Band:
    0: nodata, 1: saturated or defective, 2: cast_shadows, 3: cloud_shadows, 
    4: vegetation, 5: not_vegetated, 6: water, 7: unclassified, 8: cloud_medium_prob,
    9: cloud_high_prob, 10: thin_cirrus, 11: snow_ice
    """
    land_cover_endcoding = dict(fill=[0],
                                invalid=[1],
                                cast_shadows=[2],
                                cloud_shadows=[3],
                                vegetation=[4],
                                not_vegetated=[5],
                                water=[6],
                                unclassified=[7],
                                cloud_medium_prob=[8],
                                cloud_high_prob=[9],
                                thin_cirrus=[10],
                                snow_ice=[11],
                                clear = [4,5,6,11],
                                )
    return unpack_bits(land_cover_endcoding, data_array, cover_type)


def qa_clean_mask(pixel_qa_band, platform, cover_types):
    """
    Function originally from dc_mosaic.py
    """
    processing_options = {
        "LANDSAT_8": ls_unpack_qa,
        "SENTINEL_2": sen2_unpack_qa
    }

    clean_mask = None
    # Keep all specified cover types (e.g. 'clear', 'water'), so logically or the separate masks.
    
    if platform == "SENTINEL_2":
        for i, cover_type in enumerate(cover_types):
            cover_type_clean_mask = processing_options[platform](pixel_qa_band, cover_type)
            clean_mask = cover_type_clean_mask if i == 0 else xr_or(clean_mask, cover_type_clean_mask)
    
    else: 
        for i, cover_type in enumerate(cover_types):
            logging.info(f'running cover_type: {cover_type}')
            cover_type_clean_mask = processing_options[platform](pixel_qa_band, cover_type)
            clean_mask = cover_type_clean_mask if i == 0 else xr_or(clean_mask, cover_type_clean_mask)

    return clean_mask

def rename_bands(in_xr, des_bands, position):
    in_xr.name = des_bands[position]
    return in_xr

def load_img(bands_data, band_nms, satellite):
    """
    Assumes first band is ref

    Note: for Sentinel-2, first band has 10m resolution which causes capacity issues when running, so switch to using 
    bands_data[6] (20m resolution) as the reference band for resampling and reprojection. 
    """
    atts = bands_data[0].attrs
    bands_data = [ rename_bands(band_data, band_nms, i) for i,band_data in enumerate(bands_data) ] # rename
   
    # Pick the reference band for reprojection and resampling 
    if satellite == 'SENTINEL_2':
        ref_band = bands_data[6]
    elif satellite.startswith('LANDSAT_'):
        ref_band = bands_data[0]

    bands_data = [ bands_data[i].rio.reproject_match(ref_band) for i in range(len(band_nms)) ] # repro+resample+extent
    bands_data = [ xr.align(bands_data[0], bands_data[i], join="override")[1] for i in range(len(band_nms)) ] # force align
    bands_data = xr.merge(bands_data).rename({'band': 'time'}).isel(time = 0).drop(['time']) # ensure band names & dims consistent
    bands_data = bands_data.assign_attrs(atts)
    return bands_data

def gen_water_mask(optical_yaml_path, s3_source=False, s3_bucket='', s3_dir='common_sensing/fiji/wofsdefault/', inter_dir='test_masking/', aoi_mask=False, **kwargs):
    """
    Function to generate water masks for input data.
    """
    logging.info(f"Optical yaml path: {optical_yaml_path}")

    # Assume dirname of yml references name of the scene - should hold true for all ard-workflows prepared scenes
    scene_name = os.path.dirname(optical_yaml_path).split('/')[-1]
    logging.info(f"Scene name: {scene_name}")
    
    # Data to run the water mask on is stored in data_dir for now (should eventually pull from s3)
    data_dir = f"{inter_dir}{scene_name}/"
    os.makedirs(data_dir, exist_ok=True)

    logging.info(f"Data dir: {data_dir}")

    masked_dir = f"{inter_dir}{scene_name}_masked/"
    os.makedirs(masked_dir, exist_ok=True)
    #cog_dir = f"{inter_dir}{scene_name}/"
    #os.makedirs(cog_dir, exist_ok=True)

    # Set up logging
    root.info(f"{scene_name} starting")

    yml = optical_yaml_path
    #yml = f'{inter_dir}datacube-metadata.yaml'
    #aoi = f'{inter_dir}mask_aoi.geojson'

    # Define the desired bands for each instrument
    des_band_refs = {
        "LANDSAT_8": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "LANDSAT_7": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "LANDSAT_5": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "LANDSAT_4": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "SENTINEL_2": ['blue','green','red','nir','swir1','swir2','scene_classification'],
        "SENTINEL_1": ['VV','VH','somethinglayover shadow']}

    # Download the data and yml
    try:
        root.info(f"Downloading {scene_name}")

        # Note: haven't tested the following section (because s3_source=False)
        if (s3_source) & (not os.path.exists(yml)):
            s3_download(s3_bucket, optical_yaml_path, yml)
            with open (yml) as stream: yml_meta = yaml.safe_load(stream)
            satellite = yml_meta['platform']['code'] # helper to generalise masking 
            des_bands = des_band_refs[satellite]
            print(satellite, des_bands)
            band_paths_s3 = [os.path.dirname(optical_yaml_path)+'/'+yml_meta['image']['bands'][b]['path'] for b in des_bands ]
            band_paths_local = [inter_dir+os.path.basename(i) for i in band_paths_s3]
            for s3, loc in zip(band_paths_s3, band_paths_local): 
                if not os.path.exists(loc):
                    s3_download(s3_bucket, s3, loc)
        elif os.path.exists(yml):
            with open (yml) as stream: yml_meta = yaml.safe_load(stream)
            satellite = yml_meta['platform']['code'] # helper to generalise masking 
            root.info(f'Satellite: {satellite}')
            des_bands = des_band_refs[satellite]
        else:
            print('boo')
        
        if aoi_mask:
            #s3_download(s3_bucket, aoi_mask, aoi)
            root.info(f"Using AOI mask")
        else: 
            aoi = False
        root.info(f"Found and downloaded the yml and data")
    except:
        root.exception(f"{scene_name} Yaml or band files can't be found")
        raise Exception('Download Error')
    
    # Reformatting Bands
    try:
        root.info(f"{scene_name} Loading & Reformatting bands")

        # data loading pre-requisite xarray format for applying mask + wofs classifier
        o_bands_data = [ rxr.open_rasterio(data_dir + yml_meta['image']['bands'][b]['path']) for b in des_bands ] # loading
        satellite = yml_meta['platform']['code'] # helper to generalise masking 
        logging.info(f'Satellite: {satellite}')
        
        # Use the load_img function to resample, reproject, align and merge bands into single dataset
        bands_data = load_img(o_bands_data, des_bands, satellite)

        # Select the band from which to generate masks
        if satellite == 'SENTINEL_2':
            pixel_qa_band = bands_data.scene_classification
        elif satellite.startswith('LANDSAT_'):
            pixel_qa_band = bands_data.pixel_qa

        for i in o_bands_data: i.close()
        #o_bands_data = None
        timestamp = datetime.strptime(yml_meta['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')
        bands_data['time'] = timestamp
        root.info(f"{scene_name} Loaded & Reformatted bands")
        # log number of bands loaded
        #root.info(f"{scene_name} Loaded {len(o_bands_data.data_vars)} bands")
    except:
        root.exception(f"{scene_name} Band data not loaded properly")
        raise Exception('Data formatting error')
    
    # Generating water masks
    try:
        root.info(f"{scene_name} Applying masks")
        
        # Generate the water/clear masks and write to raster
        water_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['water']) 
        water_mask['time'] = timestamp
        water_mask.rio.to_raster(f"{masked_dir}/{scene_name}_water_mask.tif", dtype="uint8", driver='COG')

        clear_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['clear']) 
        clear_mask['time'] = timestamp
        clear_mask.rio.to_raster(f"{masked_dir}/{scene_name}_clear_mask.tif", dtype="uint8", driver='COG')

        # Combine the clear and water masks
        combined_mask = (water_mask.astype(int) + clear_mask.astype(int)) - 1 

        combined_mask['time'] = timestamp
        combined_mask.rio.to_raster(f"{masked_dir}/{scene_name}_combined_mask.tif", dtype="int8", driver='COG')
        logging.info(f"combined mask: {combined_mask}")

        root.info(f"Got the masks for {satellite}")

        root.info(f'COMBINED MASK: {combined_mask}')
        root.info(f'COMBINED MASK ATTRS: {combined_mask.attrs}')

        # Apply the clearsky mask
        root.info(f"Applying the cloud masks for {satellite}")
        
        # Loop through the bands of o_bands_data and apply the clearsky mask
        # for var in bands_data.data_vars:
        #     masked = bands_data[var].where(clear_mask)
        #     masked.rio.to_raster(f"{masked_dir}/{scene_name}_{var}_masked.tif")
        #     logging.info(f"Wrote raster for {var}")
    
    except:
        root.exception(f"{scene_name} Masks not applied")
        raise Exception('Data formatting error')
    

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    # # Landsat 8 
    #optical_yaml_path = '/home/spatialdays/Documents/ARD_Data/ARD_Landsat/ScaleValues/Scaled/LC08_L2SR_076071_20200910_tmp/LC08_L2SR_076071_20200910/datacube-metadata.yaml'
    
    # Sentinel 2
    #optical_yaml_path = '/home/spatialdays/Documents/ARD_Data/ARD_Sentinel2/S2A_MSIL2A_20190124T221941_T60KYF_tmp/S2A_MSIL2A_20190124T221941_T60KYF_scaled/datacube-metadata.yaml'
    
    #yaml_paths = glob.glob('/home/spatialdays/Documents/ARD_Data/ARD_Landsat/Tile7572/*/datacube-metadata.yaml')
    # #dirs = [os.path.dirname(dir) for dir in dirs_path]
    # #logging.info(f"Directories: {dirs}")

    #for optical_yaml_path in yaml_paths:
        #gen_water_mask(optical_yaml_path, inter_dir='test_masking/Tile7572/')

    optical_yaml_path = '/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/LC08_L2SR_075072_20221011_tmp/datacube-metadata.yaml'
    gen_water_mask(optical_yaml_path, inter_dir='test_masking/Tile7572/')

    
