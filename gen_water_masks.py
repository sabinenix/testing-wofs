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
from wofs_classify import wofs_classify


def ls_unpack_qa(data_array, cover_type):
    """
    (Originally ls8_unpack_qa, ls7_unpack_qa, etc. from dc_mosaic.py)

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

def unpack_bits(land_cover_endcoding, data_array, cover_type):
    """
	Description:
		Unpack bits for the sen2_unpack_qa function.
	-----
	Input:
		land_cover_encoding(dict hash table) land cover endcoding for the scene classification band
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

def sen2_unpack_qa(data_array, cover_type):
    """
    (Originally from dc_mosaic.py)

    Changed values to correspond to the Sentinel-2 Scene Classification Band:
    0: nodata, 1: saturated or defective, 2: cast_shadows, 3: cloud_shadows, 
    4: vegetation, 5: not_vegetated, 6: water, 7: unclassified, 8: cloud_medium_prob,
    9: cloud_high_prob, 10: thin_cirrus, 11: snow_ice

    Documentation on these values: 
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
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
    (Originally from dc_mosaic.py)

    Parameters:
    pixel_qa_band: xarray DataArray of either the pixel_qa band for Landsat or
    the scene classification band for Sentinel-2
    platform: string denoting the platform (e.g. LANDSAT_8, SENTINEL_2)
    cover_types: list of strings denoting the cover types to return mask for - 
    current options are only 'clear' or 'water'

    Returns: 
    clean_mask: xarray DataArray with a boolean mask for the cover type
    """
    processing_options = {
        "LANDSAT_4": ls_unpack_qa,
        "LANDSAT_5": ls_unpack_qa,
        "LANDSAT_7": ls_unpack_qa,
        "LANDSAT_8": ls_unpack_qa,
        "SENTINEL_2": sen2_unpack_qa
    }

    clean_mask = None
    
    if platform == "SENTINEL_2":
        for i, cover_type in enumerate(cover_types):
            cover_type_clean_mask = processing_options[platform](pixel_qa_band, cover_type)
            clean_mask = cover_type_clean_mask if i == 0 else xr_or(clean_mask, cover_type_clean_mask)
    
    else: 
        for i, cover_type in enumerate(cover_types):
            cover_type_clean_mask = processing_options[platform](pixel_qa_band, cover_type)
            clean_mask = cover_type_clean_mask if i == 0 else xr_or(clean_mask, cover_type_clean_mask)

    return clean_mask

def rename_bands(in_xr, des_bands, position):
    """
    (From genprepWater.py)
    """
    in_xr.name = des_bands[position]
    return in_xr

def load_img(bands_data, band_nms, satellite, timestamp):
    """
    (From genprepWater.py)

    Use one of the bands as a reference for reprojection and resampling (matching the resolution of
    the reference band), then align all bands before merging into a single xarray dataset. 

    Note the original function used the first band by default as the reference band for reprojecting 
    and resampling, which works for Landsat, but for Sentinel-2, the first band has a 10m resolution 
    which caused capacity issues when running, so switched to using bands_data[6] (20m resolution) as 
    the reference band. 

    Parameters:
    bands_data: list of xarray DataArrays for each band
    band_nms: list of strings of the band names
    satellite: string denoting the satellite (e.g. LANDSAT_8, SENTINEL_2)
    nodata: int denoting the nodata value to use for the bands (0 for Landsat Collection 2)
    """
    # Name the bands so they appear as named data variables in the xarray dataset
    bands_data = [ rename_bands(band_data, band_nms, i) for i,band_data in enumerate(bands_data) ] 
   
    # Pick the reference band and assign nodata value for each satellite
    if satellite == 'SENTINEL_2':
        ref_band = bands_data[6]
        nodata = -9999
    elif satellite.startswith('LANDSAT_'):
        ref_band = bands_data[0]
        nodata = 0

    # Combine the bands into xarray dataset after matching them to the reference band and aligning
    bands_data = [ bands_data[i].rio.reproject_match(ref_band, nodata = nodata) for i in range(len(band_nms)) ] 
    bands_data = [ xr.align(bands_data[0], bands_data[i], join="override")[1] for i in range(len(band_nms)) ]
    
    # Add time dimension to the data - TODO: why are we adding and then dropping?
    for i in bands_data: i['time_drop'] = timestamp 
    bands_data = xr.merge(bands_data).rename({'band':'time'}).drop('time_drop')
    
    # Add attributes from the original reference band
    attrs = ref_band.attrs
    bands_data = bands_data.assign_attrs(attrs)

    return bands_data

def gen_water_mask(optical_yaml_path, s3_source=False, s3_bucket='', s3_dir='common_sensing/fiji/wofsdefault/', inter_dir='test_masking/', aoi_mask=False, apply_masks=False, **kwargs):
    """
    Generate (and if desired, apply) water and clear masks for a given scene. 
    
    Masks are generated using two methods: 1) Using the pixel_qa band (Landsat) and scene_classification band (Sentinel-2). 
    The Landsat pixel_qa band is a bit band where each bit corresponds to some surface condition. The Sentinel-2 scene_classification
    band has values corresponding to different surface conditions. 2) Using the WOFS algorithm. The WOFS algorithm works for both 
    Landsat and Sentinel-2 values and classifies a scene into water/non-water based on a model trained on Landsat data in Australia 
    (Mueller et al. 2015). If desired, the clearsky masks can be applied to the data to generate cloud-free data.
    """

    # Assume dirname of yml references name of the scene - should hold true for all ard-workflows prepared scenes
    scene_name = os.path.dirname(optical_yaml_path).split('/')[-1]
    root.info(f"Starting scene: {scene_name}")
    
    # Data to run the water mask on is stored in data_dir for now (should eventually pull from s3)
    data_dir = f"{inter_dir}{scene_name}/"
    os.makedirs(data_dir, exist_ok=True)

    # Directory to hold the water/cloud masks and masked imagery
    masked_dir = f"{inter_dir}{scene_name}_masked/"
    os.makedirs(masked_dir, exist_ok=True)

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
    
    try:
        root.info(f"{scene_name} Loading & Reformatting bands")

        # Get the timestamp for the scene from the yaml file
        timestamp = datetime.strptime(yml_meta['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')

        # Open each band and get the satellite information from the yaml file
        o_bands_data = [ rxr.open_rasterio(data_dir + yml_meta['image']['bands'][b]['path']) for b in des_bands ] 
        satellite = yml_meta['platform']['code'] 
        
        # Use the load_img function to resample, reproject, align and merge bands into single dataset
        bands_data = load_img(o_bands_data, des_bands, satellite, timestamp)

        # Select the band from which to generate masks
        if satellite == 'SENTINEL_2':
            pixel_qa_band = bands_data.scene_classification
        elif satellite.startswith('LANDSAT_'):
            pixel_qa_band = bands_data.pixel_qa

        for i in o_bands_data: i.close()

        root.info(f"{scene_name} Loaded & Reformatted bands")

    except:
        root.exception(f"{scene_name} Band data not loaded properly")
        raise Exception('Data formatting error')
    
    try:
        root.info(f"{scene_name} Generating masks using classification bands")
        
        # Generate the water mask
        water_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['water']) 
        water_mask.rio.to_raster(f"{masked_dir}/{scene_name}_water_mask.tif", dtype="int16", driver='COG')

        # Generate the clear mask
        clear_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['clear']) 
        clear_mask.rio.to_raster(f"{masked_dir}/{scene_name}_clear_mask.tif", dtype="int16", driver='COG')

        # Combine the clear and water masks (nodata = -9999, non-water = 0, water = 1)
        combined_mask = water_mask.where(clear_mask)
        combined_mask = combined_mask.fillna(-9999)
        logging.info(f'Min and Max of combined mask: {combined_mask.min()}, {combined_mask.max()}')
        combined_mask.rio.to_raster(f"{masked_dir}/{scene_name}_combined_mask.tif", nodata=-9999, dtype="int16", driver='COG')

        root.info(f"Got the masks for {satellite}")
    
    except:
        root.exception(f"{scene_name} Masks not generated")
        raise Exception('Data formatting error')
    
    
    # Generate water masks using wofs algorithm
    try: 
        root.info(f"{scene_name} Generating masks using WOFS algorithm")
        wofl = wofs_classify(bands_data, clean_mask=clear_mask, x_coord='x', y_coord='y',
                  time_coord='time', no_data=-9999)
        logging.info(f'Min and Max of WOFL: {wofl.wofs.min()}, {wofl.wofs.max()}')
        wofl.wofs.rio.to_raster(f"{masked_dir}/{scene_name}_wofl_mask.tif", nodata=-9999, dtype="int16", driver='COG')
        root.info(f"Got the WOFL masks for {satellite}")
    except:
        root.exception(f"{scene_name} WOFL masks not generated")
        raise Exception('Data formatting error')

    
    # If specified, apply masks to generate cloud-free data
    if apply_masks:
        try: 
            root.info(f"Applying the cloud masks for {satellite}")
            
            # Loop through the bands of bands_data and apply the clearsky mask
            for var in bands_data.data_vars:
                logging.info(f"Writing raster for {var}")
                masked = bands_data[var].where(clear_mask)
                masked.rio.to_raster(f"{masked_dir}/{scene_name}_{var}_masked.tif")
        except:
            root.exception(f"{scene_name} Masks not applied")
            raise Exception('Data formatting error')

    

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    # # Landsat 8 
    #optical_yaml_path = '/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/LC08_L2SR_075072_20221011_tmp/datacube-metadata.yaml'
    #gen_water_mask(optical_yaml_path, inter_dir='test_masking/Tile7572/', apply_masks=False)


    # # Sentinel 2
    #optical_yaml_path = '/home/spatialdays/Documents/testing-wofs/test_masking/S2A_MSIL2A_20190124T221941_T60KYF_scaled_tmp/datacube-metadata.yaml'
    #gen_water_mask(optical_yaml_path, inter_dir='test_masking/')

    # Sentinel 2: Run for all scenes in the directory
    

    # Landsat 8: Run for all scenes in the directory
    yaml_paths = glob.glob('/home/spatialdays/Documents/ARD_Data/ARD_Landsat/Tile7572/*/datacube-metadata.yaml')
    for optical_yaml_path in yaml_paths:
        gen_water_mask(optical_yaml_path, inter_dir='test_masking/Tile7572/')
    
