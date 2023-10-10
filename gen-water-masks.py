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


def qa_clean_mask(pixel_qa_band, platform, cover_types):
    """
    Function originally from dc_mosaic.py
    """
    processing_options = {
        "LANDSAT_8": ls_unpack_qa,
        #"SENTINEL_2": sen2_unpack_qa
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

def gen_water_mask(optical_yaml_path, s3_source=False, s3_bucket='', s3_dir='common_sensing/fiji/wofsdefault/', inter_dir='test_masking/data/', aoi_mask=False, **kwargs):
    """
    Function to generate water masks for input data.
    """
    # Assume dirname of yml references name of the scene - should hold true for all ard-workflows prepared scenes
    scene_name = os.path.dirname(optical_yaml_path).split('/')[-1]
    
    # Make temporary directories to hold outputs
    inter_dir = f"{inter_dir}{scene_name}_tmp/"
    os.makedirs(inter_dir, exist_ok=True)
    cog_dir = f"{inter_dir}{scene_name}/"
    os.makedirs(cog_dir, exist_ok=True)

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

        # Note: haven't tested the following section
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

        o_bands_data = [ rxr.open_rasterio(inter_dir + yml_meta['image']['bands'][b]['path']) for b in des_bands ] # loading
        pixel_qa_band = o_bands_data[6]
        #o_bands_data = [ resamp_bands(i, o_bands_data) for i in o_bands_data ]

        # rescale bands using scale_and_clip_dataarray function, only not for the last item in the list
        #logging.info(f"{scene_name} Rescaling bands to old collection 1 style")
        #o_bands_data = [ scale_and_clip_dataarray(i, scale_factor=0.275, add_offset=-2000,clip_range=None, valid_range=(0, 10000))  for i in o_bands_data[:-1] ] + [o_bands_data[-1]]

        #bands_data = xr.merge([rename_bands(bd, des_bands, i) for i,bd in enumerate(o_bands_data)]).rename({'band': 'time'}) # ensure band names & dims consistent
        #bands_data = bands_data.assign_attrs(o_bands_data[0].attrs) # crs etc. needed later
        for i in o_bands_data: i.close()
        #o_bands_data = None
        #bands_data['time'] = [datetime.strptime(yml_meta['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')] # time dim needed for wofs
        root.info(f"{scene_name} Loaded & Reformatted bands")
        # log number of bands loaded
        #root.info(f"{scene_name} Loaded {len(o_bands_data.data_vars)} bands")
    except:
        root.exception(f"{scene_name} Band data not loaded properly")
        raise Exception('Data formatting error')
    
    # Generating water masks
    try:
        root.info(f"{scene_name} Applying masks")

        # Define the output paths for the water + combined (clear+water) masks
        water_mask_opath = f"{scene_name}_water_mask.tif"
        combined_mask_opath = f"{scene_name}_combined_mask.tif"
        
        if 'LANDSAT' in satellite:
            # Generate the water mask and write to raster
            water_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['water']) 
            water_mask.rio.to_raster(water_mask_opath, dtype="uint8")

            # Generate the clear (cloud-free) mask
            clear_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['clear']) 

            # Combine the clear and water masks (0: nodata/cloud; 1: clear land; 2: clear water)
            combined_mask = water_mask.astype(int) + clear_mask.astype(int)
            combined_mask.rio.to_raster(combined_mask_opath, dtype="uint8")
            
            root.info(f"Got the water mask for {satellite}")
        # elif 'SENTINEL_2' in satellite:
        #     clearsky_masks = (
        #         (bands_data.scene_classification == 2) | # DARK_AREA_PIXELS
        #         (bands_data.scene_classification == 4) | # VEGETATION
        #         (bands_data.scene_classification == 5) | # NON_VEGETATION
        #         (bands_data.scene_classification == 6) | # WATER
        #         (bands_data.scene_classification == 7)   # UNCLASSIFIED
        #     )
        else:
            raise Exception('clearsky masking not possible')
        # elif sentinel-1 in satellite:
#             clearsky_masks = landsat_qa_clean_mask(bands_data, satellite) 
            

        #logger.info(f"Starting to apply the clearsky mask for {satellite}")
        #clearsky_scenes = bands_data.where(clearsky_masks) # !!!this consumes a lot of memory!!!
        #logger.info(f"Applied the clearsky mask for {satellite}")
#             if satellite == 'SENTINEL_2':
#                 clearsky_scenes = clearsky_scenes.rename_vars({'swir_1': 'swir1', 'swir_2': 'swir2'})
    except:
        root.exception(f"{scene_name} Masks not applied")
        raise Exception('Data formatting error')
    

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))


    optical_yaml_path = '/home/spatialdays/Documents/ARD_Data/ARD_Landsat/ScaleValues/Scaled/LC08_L2SR_076071_20200910_tmp/LC08_L2SR_076071_20200910/datacube-metadata.yaml'
    gen_water_mask(optical_yaml_path)
