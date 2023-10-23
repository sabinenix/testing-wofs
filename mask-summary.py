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

def add_timestamp_data_to_xr(dataset):
    """Add timestamp data to an xarray dataset using the time dimension.

    Adds both a timestamp and a human readable date int to a dataset - int32 format required.
    modifies the dataset in place.
    """
    dims_data_var = list(dataset.data_vars)[0]

    timestamp_data = np.full(dataset[dims_data_var].values.shape, 0, dtype="int32")
    date_data = np.full(dataset[dims_data_var].values.shape, 0, dtype="int32")

    for index, acq_date in enumerate(dataset.time.values.astype('M8[ms]').tolist()):
        timestamp_data[index::] = acq_date.timestamp()
        date_data[index::] = int(acq_date.strftime("%Y%m%d"))
    dataset['timestamp'] = xr.DataArray(
        timestamp_data,
        dims=('time', 'latitude', 'longitude'),
        coords={'latitude': dataset.latitude,
                'longitude': dataset.longitude,
                'time': dataset.time})
    dataset['date'] = xr.DataArray(
        date_data,
        dims=('time', 'latitude', 'longitude'),
        coords={'latitude': dataset.latitude,
                'longitude': dataset.longitude,
                'time': dataset.time})

def perform_timeseries_analysis(dataset_in, band_name, intermediate_product=None, no_data=0, operation="mean"):
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

    assert operation in ['mean', 'max', 'min'], "Please enter a valid operation."

    logging.info(f'starting timeseries analysis')

    data = dataset_in
    data = data.where(data != no_data)

    processed_data_sum = data.sum('time')
    logging.info(f'PROCESSED DATA SUM: {processed_data_sum}')

    clean_data = data.notnull()

    clean_data_sum = clean_data.astype('bool').sum('time')

    dataset_out = None
    if intermediate_product is None:
        processed_data_normalized = processed_data_sum / clean_data_sum
        dataset_out = xr.Dataset(
            {
                'normalized_data': processed_data_normalized,
                'min': data.min(dim='time'),
                'max': data.max(dim='time'),
                'total_data': processed_data_sum,
                'total_clean': clean_data_sum.astype('int8')
            },
            coords={'latitude': dataset_in.x,
                    'longitude': dataset_in.y})
    else:
        dataset_out = intermediate_product
        dataset_out['total_data'] += processed_data_sum
        dataset_out['total_clean'] += clean_data_sum.astype('int8')
        dataset_out['normalized_data'] = dataset_out['total_data'] / dataset_out['total_clean']
        dataset_out['min'] = xr.concat([dataset_out['min'], data.min(dim='time')], dim='time').min(dim='time')
        dataset_out['max'] = xr.concat([dataset_out['max'], data.max(dim='time')], dim='time').max(dim='time')

    nan_to_num(dataset_out, 0)

    return dataset_out


def mask_summary(mask_dir='/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/*_masked/*_combined_mask.tif'):
    """
    Combine 'WOFS'-like layers into a 'WOFS SUMMARY' -like layer
    """

    # Get list of water masks (one for each scene)
    mask_paths = glob.glob(mask_dir)

    scenes = []

    for mask in mask_paths:
        scene_name = (os.path.dirname(mask).split('/')[-1]).split('_')[0:4]
        scene_name = '_'.join(scene_name)

        logging.info(f'SCENE NAME: {scene_name}')

        # Set all pixels with values of 0 to nan so they are excluded from clean data sum
        #mask.values[mask.values == 0] = np.nan

        # TODO: Don't hard code this 
        yaml_path = f'/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/{scene_name}_tmp/datacube-metadata.yaml'

        # Get the timestamp from the yaml file 
        with open (yaml_path) as stream: yml_meta = yaml.safe_load(stream)
        timestamp = datetime.strptime(yml_meta['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')
        mask = rxr.open_rasterio(mask)
        mask['time'] = timestamp
        scenes.append(mask)     
        # Close the mask files
        mask.close()


    # Match res/projection and force align so the scenes can be concatenated
    scenes = [ scenes[i].rio.reproject_match(scenes[0]) for i in range(len(scenes)) ] 
    
    # Avoid coord differences due to floats (without this the code crashes bc of misaligned coords)
    scenes = [scenes[i].assign_coords({
    "x": scenes[0].x,
    "y": scenes[0].y,}) for i in range(len(scenes))]
    
    # Align the scenes so they can be concatenated later
    scenes = [ xr.align(scenes[0], scenes[i], join="override", fill_value=0)[1] for i in range(len(scenes)) ] 

    # Concatenate xarrays into single dataset
    mask_data = xr.concat(scenes, dim='time', fill_value=0).rename({'band': 'vals'})
    for i in scenes: i.close()
    logging.info(f'CONCATENATED: {mask_data}')



    # Run timeseries analysis 
    out_data = perform_timeseries_analysis(mask_data, 'vals', intermediate_product=None, no_data=0, operation="mean")
    logging.info(f'OUT DATA: {out_data}')

    # Write out_data to tif
    out_data.normalized_data.rio.to_raster(f'/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/NormalizedData.tif')
    out_data.total_clean.rio.to_raster(f'/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/TotalClean.tif')
    out_data.total_data.rio.to_raster(f'/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/TotalData.tif')

if __name__ == '__main__':  
    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    mask_summary(mask_dir='/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/*_masked/*_combined_mask.tif')




