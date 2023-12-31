import numpy as np
import xarray as xr
import rioxarray as rxr
from osgeo import gdal
from xarray.ufuncs import logical_or  as xr_or
import yaml
from datetime import datetime
import logging
import os
import glob
from wofs_classify import wofs_classify
import botocore
import boto3
from dateutil.parser import parse
import uuid


def ls_unpack_qa(data_array, cover_type):
    """
    (Originally ls8_unpack_qa, ls7_unpack_qa, etc. from dc_mosaic.py)

    For l4, l5, l7, and l8, the pixel_qa band is a bit band where each bit 
    corresponds to some surface condition. The 6th bit corresponds to clear, 
    and the 7th bit corresponds to water. Note that a pixel will only be flagged
    as water if it is also not cloud, cloud shadow, etc.

    For l4-7, clear bit is a known issue and USGS recommends using bits 1 and 3.
    (https://www.usgs.gov/landsat-missions/landsat-collection-2-known-issues#SR)

    Parameters: 
    data_array - xarray DataArray of the scene's pixel_qa band
    cover_type - string of the cover type to return mask for (optiosn are 'clear' or 'water')

    Returns: 
    boolean_mask - xarray DataArray with a boolean mask for the cover type (True where pixels
    are clear or water, and False where not clear or not water, depending on the mask type)
    """
    boolean_mask = np.zeros(data_array.shape, dtype=bool)
    data_array = data_array.astype(np.int64)

    # Create a mask for nodata areas around edges of the image (value of 1 in pixel_qa band)
    nodata_mask = data_array == 1

    if cover_type == 'clear':
        # If dilated cloud (1), cloud (3), cloud shadow (4) and snow (5) bits are OFF, pixel is clear
        dilated_cloud_bit = 1 << 1
        cloud_bit = 1 << 3
        cloud_shadow_bit = 1 << 4
        snow_bit = 1 << 5
        clear_bit_mask = dilated_cloud_bit | cloud_bit | cloud_shadow_bit | snow_bit
        boolean_mask |= np.bitwise_and(data_array, clear_bit_mask) == 0

        # Apply the nodata mask to force edges of image to have value of 0 
        boolean_mask = boolean_mask & ~nodata_mask
    
    elif cover_type == 'water':
        # 7th bit is water
        boolean_mask |= ((data_array & 0b0000000010000000) != 0)
    else:
        raise ValueError(f"Cover type {cover_type} not supported for Landsat 8 yet")

    return boolean_mask

def unpack_bits(land_cover_endcoding, data_array, cover_type):
    """
    (From dc_mosaic.py)
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

    Clear masks include all pixels that are not nodata, saturated or defected, cloud shadows, 
    clouds (medium and high prob), or thin cirrus.

    Documentation on these values: 
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
    
    Parameters:
    data_array - xarray DataArray of the scene's scene_classification band
    cover_type - string of the cover type to return mask for (options are 'clear' or 'water')

    Returns:
    boolean_mask - xarray DataArray with a boolean mask for the cover type (True where pixels
    are clear or water, and False where not clear or not water, depending on the mask type)
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
                                clear = [2,4,5,6,7,11],
                                )
    return unpack_bits(land_cover_endcoding, data_array, cover_type)


def qa_clean_mask(pixel_qa_band, platform, cover_types):
    """
    (Originally from dc_mosaic.py)

    Parameters:
    pixel_qa_band - xarray DataArray of either the pixel_qa band for Landsat or
    the scene classification band for Sentinel-2
    platform - string denoting the platform (e.g. LANDSAT_8, SENTINEL_2)
    cover_types - list of strings denoting the cover types to return mask for - 
    current options are only 'clear' or 'water'

    Returns: 
    clean_mask - xarray DataArray with a boolean mask for the cover type
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

def load_img(bands_data, band_nms, satellite):
    """
    (Originally from genprepWater.py)

    Use one of the bands as a reference for reprojection and resampling (matching the resolution of
    the reference band), then align all bands before merging into a single xarray dataset. 

    Note the original function used the first band by default as the reference band for reprojecting 
    and resampling, which works for Landsat, but for Sentinel-2, the first band has a 10m resolution 
    which caused capacity issues when running, so switched to using bands_data[6] (20m resolution) as 
    the reference band. 

    Parameters:
    bands_data - list of xarray DataArrays for each band
    band_nms- list of strings of the band names
    satellite -  string denoting the satellite (e.g. LANDSAT_8, SENTINEL_2)

    Returns:
    bands_data - xarray dataset containing all bands
    """
    # Name the bands so they appear as named data variables in the xarray dataset
    bands_data = [ rename_bands(band_data, band_nms, i) for i,band_data in enumerate(bands_data) ] 
   
    # Pick the reference band for each satellite
    if satellite == 'SENTINEL_2':
        ref_band = bands_data[6]
    elif satellite.startswith('LANDSAT_'):
        ref_band = bands_data[0]

    # Combine the bands into xarray dataset after matching them to the reference band and aligning
    bands_data = [ bands_data[i].rio.reproject_match(ref_band) for i in range(len(band_nms)) ] 
    bands_data = [ xr.align(bands_data[0], bands_data[i], join="override")[1] for i in range(len(band_nms)) ]
    
    # Rename band to 'time' because wofs expects a time dimension (even though it's not a proper timestamp)
    bands_data = xr.merge(bands_data).rename({'band':'time'})
    
    # Add attributes from the original reference band
    attrs = ref_band.attrs
    bands_data = bands_data.assign_attrs(attrs)

    # Add fill value attribute to -9999
    bands_data.attrs['_FillValue'] = -9999

    logging.info(f'Combined and stacked band data: {bands_data}')

    return bands_data

def s3_download(s3_bucket, s3_obj_path, dest_path):
    """ - tested only for S3"""
    client, bucket = s3_create_client(s3_bucket)
    
    try:
        bucket.download_file(s3_obj_path, dest_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logging.info("The object does not exist.")
            raise
        else:
            raise


def s3_create_client(s3_bucket):
    """
    Create and set up a connection to S3
    :param s3_bucket:
    :return: the s3 client object.
    """

    access = os.getenv("AWS_ACCESS_KEY_ID", 'none')
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", 'none')

    session = boto3.Session(
        access,
        secret,
    )

    endpoint_url = os.getenv("S3_ENDPOINT", 'http://localhost:30003')

    if endpoint_url is not None:
        logging.debug('Endpoint URL: {}'.format(endpoint_url))

    if endpoint_url is not None:
        s3 = session.resource('s3', endpoint_url=endpoint_url)
    else:
        s3 = session.resource('s3', region_name='eu-west-2')

    bucket = s3.Bucket(s3_bucket)

    if endpoint_url is not None:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            endpoint_url=endpoint_url
        )
    else:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access,
            aws_secret_access_key=secret
        )

    return s3_client, bucket

def yaml_prep_wofs(scene_dir, original_yml):
    """
    (From genprepWater.py)
    Generate a yaml for the water mask. Most of the yaml information is copied
    directly from the original scene's yaml, but the path to the product and the 
    product name are changed to refer to the water mask. 
    """
    scene_name = scene_dir.split('/')[-2]
    logging.info(f"Preparing scene {scene_name}")
    logging.info(f"Scene path {scene_dir}")
    
    # date time assumed eqv for start and stop - this isn't true and could be 
    # pulled from .xml file (or scene dir) not done yet for sake of progression
    t0=parse(str(datetime.strptime(original_yml['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')))
    t1=t0

    # Select the water mask to create the yaml for (the combined water + cloud mask)
    mask_path = glob.glob(scene_dir + '*combined_mask.tif')

    # Set the image path to the water mask
    images = { 'water': { 'path': str(mask_path[0].split('/')[-1]) } }
    
    # Set the new ID for the water mask
    new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{scene_name}_water"))
    
    # Combine the yaml information and call the product 'water_mask' instead of wofs
    return {
        'id': new_id,
        'processing_level': original_yml['processing_level'],
        'product_type': "water_mask",
        'creation_dt': str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')),
        'platform': {  
            'code': original_yml['platform']['code']
        },
        'instrument': {
            'name': original_yml['instrument']['name']
        },
        'extent': {
            'coord': original_yml['extent']['coord'],
            'from_dt': str(t0),
            'to_dt': str(t1),
            'center_dt': str(t0 + (t1 - t0) / 2)
        },
        'format': {
            'name': 'GeoTiff'
        },
        'grid_spatial': {
            'projection': original_yml['grid_spatial']['projection']
        },
        'image': {
            'bands': images
        },
        'lineage': {
            'source_datasets': original_yml['lineage']['source_datasets'],
        }  

    }

def create_yaml(scene_dir, metadata):
    """
    (From prep_utils.py)
    Create yaml for single scene directory containing cogs.
    """

    logging.info(f'METADATA FOR YAML: {metadata}')

    if scene_dir[-1] != '/':
        scene_dir = scene_dir + '/'
    yaml_path = str(scene_dir + 'datacube-metadata.yaml')  # referenece E or W in the path
    logging.info(f'YAML PATH: {yaml_path}')

    # not sure why default_flow_style is now required - strange...
    with open(yaml_path, 'w') as stream:
        yaml.dump(metadata, stream, default_flow_style=False)

    logging.debug('Created yaml: {}'.format(yaml_path))


def gen_water_mask(optical_yaml_path, s3_source=False, s3_bucket='ard-bucket', s3_dir='', inter_dir='', apply_masks=False, **kwargs):
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
    
    # Data to run the water mask on is stored in data_dir 
    data_dir = f"{inter_dir}{scene_name}/"
    os.makedirs(data_dir, exist_ok=True)

    # Directory to hold the water/cloud masks and masked imagery
    masked_dir = f"{inter_dir}{scene_name}_masked_v2/"
    os.makedirs(masked_dir, exist_ok=True)

    # Path where the yaml file should be downloaded
    yml = f'{data_dir}datacube-metadata.yaml'

    # Define the desired bands for each instrument
    des_band_refs = {
        "LANDSAT_8": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "LANDSAT_7": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "LANDSAT_5": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "LANDSAT_4": ['blue','green','red','nir','swir1','swir2','pixel_qa'],
        "SENTINEL_2": ['blue','green','red','nir','swir1','swir2','scene_classification'],
        "SENTINEL_1": ['VV','VH','somethinglayover shadow']}

    try:
        # If the yaml file doesn't already exist locally, download the ARD data from Azure
        if (s3_source) & (not os.path.exists(yml)):
            
            # Download the yaml file for the scene and write to 'yml' location
            s3_download(s3_bucket, optical_yaml_path, yml)

            # Open the yaml file and get the information on data to download
            with open (yml) as stream: yml_meta = yaml.safe_load(stream)
            satellite = yml_meta['platform']['code'] 
            des_bands = des_band_refs[satellite]

            logging.info(f'Starting to download data for {satellite} bands {des_bands}')

            # Define band paths on Azure and where to write to locally
            band_paths_s3 = [os.path.dirname(optical_yaml_path)+'/'+yml_meta['image']['bands'][b]['path'] for b in des_bands ]
            band_paths_local = [data_dir+os.path.basename(i) for i in band_paths_s3]

            # Download the data for each band and write to 'data_dir' location
            for s3, loc in zip(band_paths_s3, band_paths_local): 
                if not os.path.exists(loc):
                    s3_download(s3_bucket, s3, loc)
        
        # If the yaml already exists, skip downloading and just open the data 
        elif os.path.exists(yml):
            logging.info(f'Yaml already exists for {scene_name} - not re-downloading.')
            with open (yml) as stream: yml_meta = yaml.safe_load(stream)
            satellite = yml_meta['platform']['code'] 
            logging.info(f'Satellite: {satellite}')
            des_bands = des_band_refs[satellite]

        logging.info(f"Found or downloaded the yml and data")
    except:
        logging.exception(f"{scene_name} Yaml or band files can't be found")
        raise Exception('Download Error')
    
    try:
        root.info(f"Loading & Reformatting bands for {scene_name}")

        # Open each band and get the satellite information from the yaml file
        o_bands_data = [ rxr.open_rasterio(data_dir + yml_meta['image']['bands'][b]['path']) for b in des_bands ] 
        satellite = yml_meta['platform']['code'] 
        
        # Use the load_img function to resample, reproject, align and merge bands into single dataset
        bands_data = load_img(o_bands_data, des_bands, satellite)

        # Select the band from which to generate masks
        if satellite == 'SENTINEL_2':
            pixel_qa_band = bands_data.scene_classification
        elif satellite.startswith('LANDSAT_'):
            pixel_qa_band = bands_data.pixel_qa

        for i in o_bands_data: i.close()

        root.info(f"=Loaded & Reformatted bands")

    except:
        root.exception(f"{scene_name} Band data not loaded properly")
        raise Exception('Data formatting error')
    
    # Generate clear and water masks from the pixel_qa or scene_classification bands
    try:
        root.info(f"Generating masks using classification bands for {scene_name} ")
        
        # Generate the water mask
        water_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['water']) 
        water_mask.rio.to_raster(f"{masked_dir}/{scene_name}_water_mask.tif", dtype="int16", driver='COG')

        # Generate the clear mask
        clear_mask = qa_clean_mask(pixel_qa_band, satellite, cover_types=['clear']) 
        clear_mask.rio.to_raster(f"{masked_dir}/{scene_name}_clear_mask.tif", dtype="int16", driver='COG')

        # Combine the clear and water masks (nodata = -9999, non-water = 0, water = 1)
        combined_mask = water_mask.where(clear_mask)
        combined_mask = combined_mask.fillna(-9999)
        combined_mask.rio.to_raster(f"{masked_dir}/{scene_name}_combined_mask.tif", nodata=-9999, dtype="int16", driver='COG')

        root.info(f"Got the masks for {satellite}")
    
    except:
        root.exception(f"{scene_name} Masks not generated")
        raise Exception('Data formatting error')
    
    
    # Generate water masks using wofs algorithm
    try: 
        root.info(f"Generating masks using WOFS algorithm for {scene_name}")
        wofl = wofs_classify(bands_data, clean_mask=clear_mask, x_coord='x', y_coord='y',
                  time_coord='time', no_data=-9999)
        wofl.wofs.rio.to_raster(f"{masked_dir}/{scene_name}_wofl_mask.tif", nodata=-9999, dtype="int16", driver='COG')
        root.info(f"Got the WOFL masks for {satellite}")
    except:
        root.exception(f"{scene_name} WOFL masks not generated")
        raise Exception('Data formatting error')

    # Generate YAMLS for the water masks 
    try: 
        logging.info(f'Generating YAML for {scene_name}')
        create_yaml(masked_dir, yaml_prep_wofs(masked_dir, yml_meta))
        logging.info(f'YAML generated for {scene_name}')
    except: 
        logging.info(f'YAML not generated for {scene_name}')

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

    # Running on ARD data from azure (change the optical_yaml_path to the scene you want to run)
    s3_bucket = 'ard-bucket'
    optical_yaml_path = 'common_sensing/fiji/landsat_8/LC08_L2SR_075072_20180829/datacube-metadata.yaml'
    gen_water_mask(optical_yaml_path, s3_source=True, inter_dir='test_masking/Tile7572/', apply_masks=False)
    


    # Running locally for all scenes in a directory
    # yaml_paths = glob.glob('/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/*/datacube-metadata.yaml')
    # logging.info(f'YAML PATHS: {yaml_paths}')
    # for optical_yaml_path in yaml_paths:
    #     gen_water_mask(optical_yaml_path, s3_source=False, inter_dir='test_masking/Tile7572/')
    
