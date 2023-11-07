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

def nan_to_num(data, number):
    """
    Converts all nan values in `data` to `number`.

    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
    """
    if isinstance(data, xr.Dataset):
        for key in list(data.data_vars):
            data[key].values[np.isnan(data[key].values)] = number
    elif isinstance(data, xr.DataArray):
        data.values[np.isnan(data.values)] = number

def perform_timeseries_analysis(dataset_in, band_name, intermediate_product=None, no_data=-9999):
    """
    Description:

    -----
    Input:
      dataset_in (xarray.DataSet) - dataset with one variable to perform timeseries on
      band_name: name of the band to create stats for.
      intermediate_product: result of this function for previous data, to be combined here
    Output:
      dataset_out (xarray.DataSet) - dataset containing
        variables: normalized_data, total_data, total_clean
    """

    data = dataset_in

    # Ignore nodata values in the data
    data = data.where(data != no_data)

    # Calculate the sum of the data (ignoring nodata) at each pixel location
    processed_data_sum = data.sum('time')

    # Calculate the number of clean pixels at each pixel location 
    clean_data = data.notnull()
    clean_data_sum = clean_data.astype('bool').sum('time')

    dataset_out = None
    if intermediate_product is None:
        # Calculate the normalized data (likelihood of a pixel being water)
        processed_data_normalized = processed_data_sum / clean_data_sum
        dataset_out = xr.Dataset(
            {
                'normalized_data': processed_data_normalized.astype('float32'),
                'min': data.min(dim='time'),
                'max': data.max(dim='time'),
                'total_data': processed_data_sum.astype('int16'),
                'total_clean': clean_data_sum.astype('int16')
            },
            coords={'latitude': dataset_in.x,
                    'longitude': dataset_in.y})
    else:
        dataset_out = intermediate_product
        dataset_out['total_data'] += processed_data_sum
        dataset_out['total_clean'] += clean_data_sum.astype('int16')
        dataset_out['normalized_data'] = dataset_out['total_data'] / dataset_out['total_clean']
        dataset_out['min'] = xr.concat([dataset_out['min'], data.min(dim='time')], dim='time').min(dim='time')
        dataset_out['max'] = xr.concat([dataset_out['max'], data.max(dim='time')], dim='time').max(dim='time')

    # Nodata should have value of -9999 in output dataset
    nan_to_num(dataset_out, -9999)

    return dataset_out


def stack_masks(mask_dir):
    """
    Stack masks into a single xarray dataset for use in timeseries analysis.

    Using time data from the original scene yamls.
    """
    # Get list of water masks (one for each scene)
    mask_paths = glob.glob(mask_dir)
    logging.info(f'MASK PATHS: {mask_paths}')

    scenes = []

    # Open each mask, add timestamp and adjust fill value before adding to list of scenes
    for mask in mask_paths:
        scene_name = (os.path.dirname(mask).split('/')[-1]).split('_')[0:4]
        scene_name = '_'.join(scene_name)

        # TODO: Don't hard code this - reference the yaml for the mask's scene
        yaml_path = f'/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/{scene_name}_tmp/datacube-metadata.yaml'

        # Get the timestamp from the yaml file 
        with open (yaml_path) as stream: yml_meta = yaml.safe_load(stream)
        timestamp = datetime.strptime(yml_meta['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')
        mask = rxr.open_rasterio(mask)
        mask['time'] = timestamp
        mask = mask.assign_attrs({'_FillValue': -9999})

        scenes.append(mask)     
        mask.close()

    # Match res/projection and force align so the scenes can be concatenated
    scenes = [ scenes[i].rio.reproject_match(scenes[0], fill_value=-9999) for i in range(len(scenes)) ] 
    
    # Avoid coord differences due to floats (without this the code crashes bc of misaligned coords)
    scenes = [scenes[i].assign_coords({
    "x": scenes[0].x,
    "y": scenes[0].y,}) for i in range(len(scenes))]
    
    # Align the scenes so they can be concatenated later
    scenes = [ xr.align(scenes[0], scenes[i], join="override", fill_value=-9999)[1] for i in range(len(scenes)) ] 

    # Concatenate xarrays into single dataset
    mask_data = xr.concat(scenes, dim='time', fill_value=-9999).rename({'band': 'vals'}).drop('vals')
    for i in scenes: i.close()
    
    return mask_data


def mask_summary(mask_dir, out_dir, summary_mask='combined'):
    """
    Combine 'WOFS'-like layers into a 'WOFS SUMMARY' -like layer
    """
    # Determine the paths for the summary mask ('water', 'clear', or 'combined')
    if summary_mask == 'water':
        mask_dir = os.path.join(mask_dir, '*/*water_mask.tif')
    elif summary_mask == 'clear':
        mask_dir = os.path.join(mask_dir, '*/*clear_mask.tif')
    elif summary_mask == 'combined':
        mask_dir = os.path.join(mask_dir, '*/*combined_mask.tif')
    elif summary_mask == 'wofs':
        mask_dir = os.path.join(mask_dir, '*/*wofl_mask.tif')

    # Stack the masks in mask_dir into single xarray dataset
    mask_data = stack_masks(mask_dir) 
    logging.info(f'MASK DATA: {mask_data}')   
    
    # Run timeseries analysis 
    logging.info(f'starting timeseries analysis')
    out_data = perform_timeseries_analysis(mask_data, 'vals', intermediate_product=None, no_data=-9999)
    logging.info(f'completed timeseries analysis: {out_data}')

    # Write out_data to tif
    out_data.normalized_data.rio.to_raster(f'{out_dir}NormalizedData_{summary_mask}.tif')
    out_data.total_clean.rio.to_raster(f'{out_dir}TotalClean_{summary_mask}.tif')
    out_data.total_data.rio.to_raster(f'{out_dir}TotalData_{summary_mask}.tif')

if __name__ == '__main__':  
    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    mask_dir = '/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/Masks/'
    out_dir = f'/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/Summaries/'

    mask_summary(mask_dir, out_dir, summary_mask='wofs')




