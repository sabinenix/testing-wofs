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


def wofs_classify(dataset_in, clean_mask=None, x_coord='longitude', y_coord='latitude',
                  time_coord='time', no_data=0, mosaic=False, enforce_float64=False):
    """
    Description:
      Performs WOfS algorithm on given dataset.
    Assumption:
      - The WOfS algorithm is defined for Landsat 5/Landsat 7
    References:
      - Mueller, et al. (2015) "Water observations from space: Mapping surface water from
        25 years of Landsat imagery across Australia." Remote Sensing of Environment.
      - https://github.com/GeoscienceAustralia/eo-tools/blob/stable/eotools/water_classifier.py
    -----
    Inputs:
      dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube; should contain
        coordinates: time, latitude, longitude
        variables: blue, green, red, nir, swir1, swir2
    x_coord, y_coord, time_coord: (str) - Names of DataArrays in `dataset_in` to use as x, y,
        and time coordinates.
    Optional Inputs:
      clean_mask (nd numpy array with dtype boolean) - true for values user considers clean;
        if user does not provide a clean mask, all values will be considered clean
      no_data (int/float) - no data pixel value; default: -9999
      mosaic (boolean) - flag to indicate if dataset_in is a mosaic. If mosaic = False, dataset_in
        should have a time coordinate and wofs will run over each time slice; otherwise, dataset_in
        should not have a time coordinate and wofs will run over the single mosaicked image
      enforce_float64 (boolean) - flag to indicate whether or not to enforce float64 calculations;
        will use float32 if false
    Output:
      dataset_out (xarray.DataArray) - wofs water classification results: 0 - not water; 1 - water
    Throws:
        ValueError - if dataset_in is an empty xarray.Dataset.
    """

    def _band_ratio(a, b):
        """
        Calculates a normalized ratio index
        """
        return (a - b) / (a + b)

    def _run_regression(band1, band2, band3, band4, band5, band7):
        """
        Regression analysis based on Australia's training data
        TODO: Return type
        """

        # Compute normalized ratio indices
        ndi_52 = _band_ratio(band5, band2)
        ndi_43 = _band_ratio(band4, band3)
        ndi_72 = _band_ratio(band7, band2)

        #classified = np.ones(shape, dtype='uint8')

        classified = np.full(shape, no_data, dtype='uint8')

        # Start with the tree's left branch, finishing nodes as needed

        # Left branch
        r1 = ndi_52 <= -0.01

        r2 = band1 <= 2083.5
        classified[r1 & ~r2] = 0  #Node 3

        r3 = band7 <= 323.5
        _tmp = r1 & r2
        _tmp2 = _tmp & r3
        _tmp &= ~r3

        r4 = ndi_43 <= 0.61
        classified[_tmp2 & r4] = 1  #Node 6
        classified[_tmp2 & ~r4] = 0  #Node 7

        r5 = band1 <= 1400.5
        _tmp2 = _tmp & ~r5

        r6 = ndi_43 <= -0.01
        classified[_tmp2 & r6] = 1  #Node 10
        classified[_tmp2 & ~r6] = 0  #Node 11

        _tmp &= r5

        r7 = ndi_72 <= -0.23
        _tmp2 = _tmp & ~r7

        r8 = band1 <= 379
        classified[_tmp2 & r8] = 1  #Node 14
        classified[_tmp2 & ~r8] = 0  #Node 15

        _tmp &= r7

        r9 = ndi_43 <= 0.22
        classified[_tmp & r9] = 1  #Node 17
        _tmp &= ~r9

        r10 = band1 <= 473
        classified[_tmp & r10] = 1  #Node 19
        classified[_tmp & ~r10] = 0  #Node 20

        # Left branch complete; cleanup
        del r2, r3, r4, r5, r6, r7, r8, r9, r10
        gc.collect()

        # Right branch of regression tree
        r1 = ~r1

        r11 = ndi_52 <= 0.23
        _tmp = r1 & r11

        r12 = band1 <= 334.5
        _tmp2 = _tmp & ~r12
        classified[_tmp2] = 0  #Node 23

        _tmp &= r12

        r13 = ndi_43 <= 0.54
        _tmp2 = _tmp & ~r13
        classified[_tmp2] = 0  #Node 25

        _tmp &= r13

        r14 = ndi_52 <= 0.12
        _tmp2 = _tmp & r14
        classified[_tmp2] = 1  #Node 27

        _tmp &= ~r14

        r15 = band3 <= 364.5
        _tmp2 = _tmp & r15

        r16 = band1 <= 129.5
        classified[_tmp2 & r16] = 1  #Node 31
        classified[_tmp2 & ~r16] = 0  #Node 32

        _tmp &= ~r15

        r17 = band1 <= 300.5
        _tmp2 = _tmp & ~r17
        _tmp &= r17
        classified[_tmp] = 1  #Node 33
        classified[_tmp2] = 0  #Node 34

        _tmp = r1 & ~r11

        r18 = ndi_52 <= 0.34
        classified[_tmp & ~r18] = 0  #Node 36
        _tmp &= r18

        r19 = band1 <= 249.5
        classified[_tmp & ~r19] = 0  #Node 38
        _tmp &= r19

        r20 = ndi_43 <= 0.45
        classified[_tmp & ~r20] = 0  #Node 40
        _tmp &= r20

        r21 = band3 <= 364.5
        classified[_tmp & ~r21] = 0  #Node 42
        _tmp &= r21

        r22 = band1 <= 129.5
        classified[_tmp & r22] = 1  #Node 44
        classified[_tmp & ~r22] = 0  #Node 45

        # Completed regression tree

        return classified

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)
    
    # Extract dataset bands needed for calculations
    blue = dataset_in.blue
    green = dataset_in.green
    red = dataset_in.red
    nir = dataset_in.nir
    swir1 = dataset_in.swir1
    swir2 = dataset_in.swir2

    # Enforce float calculations - float64 if user specified, otherwise float32 will do
    dtype = blue.values.dtype  # This assumes all dataset bands will have
    # the same dtype (should be a reasonable
    # assumption)

    # Save dtypes because the `astype()` calls below modify `dataset_in`.
    band_list = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype

    if enforce_float64:
        if dtype != 'float64':
            blue.values = blue.values.astype('float64')
            green.values = green.values.astype('float64')
            red.values = red.values.astype('float64')
            nir.values = nir.values.astype('float64')
            swir1.values = swir1.values.astype('float64')
            swir2.values = swir2.values.astype('float64')
    else:
        if dtype == 'float64':
            pass
        elif dtype != 'float32':
            blue.values = blue.values.astype('float32')
            green.values = green.values.astype('float32')
            red.values = red.values.astype('float32')
            nir.values = nir.values.astype('float32')
            swir1.values = swir1.values.astype('float32')
            swir2.values = swir2.values.astype('float32')

    shape = blue.values.shape
    classified = _run_regression(blue.values, green.values, red.values, nir.values, swir1.values, swir2.values)

    classified_clean = np.full(classified.shape, no_data, dtype='float64')
    classified_clean[clean_mask] = classified[clean_mask]  # Contains data for clear pixels

    # Create xarray of data
    x_coords = dataset_in[x_coord]
    y_coords = dataset_in[y_coord]

    time = None
    coords = None
    dims = None

    if mosaic:
        coords = [y_coords, x_coords]
        dims = [y_coord, x_coord]
    else:
        time_coords = dataset_in[time_coord]
        coords = [time_coords, y_coords, x_coords]
        dims = [time_coord, y_coord, x_coord]

    data_array = xr.DataArray(classified_clean, coords=coords, dims=dims)

    if mosaic:
        dataset_out = xr.Dataset({'wofs': data_array},
                                 coords={y_coord: y_coords, x_coord: x_coords})
    else:
        dataset_out = xr.Dataset(
            {'wofs': data_array},
            coords={time_coord: time_coords, y_coord: y_coords, x_coord: x_coords})

    # Handle datatype conversions.
    #restore_or_convert_dtypes(None, band_list, dataset_in_dtypes, dataset_in, no_data)
    return dataset_out

def rename_bands(in_xr, des_bands, position):
    in_xr.name = des_bands[position]
    return in_xr

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

def stack_bands(scene_dir):
    # Get list of water masks (one for each scene)
    band_paths = glob.glob(f'{scene_dir}*.tif')
    logging.info(f'BAND PATHS: {band_paths}')

    # Split the file name to figure out the product
    bands = []

    for band in band_paths:
        file_name = band.split('/')[-1]
        file_parts = file_name.split('_')
        #logging.info(f'FILE PARTS: {file_parts}')
        prod_name = f"{file_parts[-2]}_{file_parts[-1][:-4]}".lower()

        #logging.info(f'BAND NAME: {prod_name}')

        prod_map = {
        "sr_b1": 'blue',
        "sr_b2": 'green',
        "sr_b3": 'red',
        "sr_b4": 'nir',
        "sr_b5": 'swir1',
        "sr_b6": 'thermal',
        "sr_b7": 'swir2',
        "qa_radsat": 'radsat_qa',
        "qa_pixel": 'pixel_qa',
        "qa_aerosol": 'aerosol_qa'
        }

        if prod_name.startswith('sr_b'):
            # TODO: Don't hard code this 
            yaml_path = f'{scene_dir}datacube-metadata.yaml'

            # Get the timestamp from the yaml file 
            with open (yaml_path) as stream: yml_meta = yaml.safe_load(stream)
            timestamp = datetime.strptime(yml_meta['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')
            band = rxr.open_rasterio(band)
            band.name = prod_map[prod_name]

            band['timee'] = timestamp
            bands.append(band)     
            # Close the mask files
            band.close()

    # Match res/projection and force align so the scenes can be concatenated
    bands = [ bands[i].rio.reproject_match(bands[0]) for i in range(len(bands)) ] 
    
    # Avoid coord differences due to floats (without this the code crashes bc of misaligned coords)
    bands = [bands[i].assign_coords({
    "x": bands[0].x,
    "y": bands[0].y,}) for i in range(len(bands))]
    
    # Align the scenes so they can be concatenated later
    bands = [ xr.align(bands[0], bands[i], join="override", fill_value=0)[1] for i in range(len(bands)) ] 
    logging.info(f'ALIGNED: {bands}')

    # Concatenate xarrays into single dataset
    #band_data = xr.concat(bands, dim='time', fill_value=0).rename({'band': 'vals'})
    band_data = xr.merge(bands).rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'}).drop('timee')
    for i in bands: i.close()
    logging.info(f'MERGED: {band_data}')

    return band_data



def clear_mask_to_boolean(clear_mask):
    # Read in the clear mask
    clear_mask = rxr.open_rasterio(clear_mask)

    # Convert clear_mask to boolean
    clear_mask = clear_mask.astype('bool')

    return clear_mask

def wofs_summary(wofl_dir='/home/spatialdays/Documents/testing-wofs/WOFLs/*.tif'):
    """
    Combine 'WOFS'-like layers into a 'WOFS SUMMARY' -like layer
    """

    # Get list of water masks (one for each scene)
    wofl_paths = glob.glob(wofl_dir)
    logging.info(f'WOFL PATHS: {wofl_paths}')

    wofls = []

    for wofl in wofl_paths:
        scene_name = (os.path.dirname(wofl).split('/')[-1]).split('_')[0:4]
        scene_name = '_'.join(scene_name)

        logging.info(f'SCENE NAME: {scene_name}')

        # Set all pixels with values of 0 to nan so they are excluded from clean data sum
        #mask.values[mask.values == 0] = np.nan

        # TODO: Don't hard code this 
        #yaml_path = f'/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/{scene_name}_tmp/datacube-metadata.yaml'

        # Get the timestamp from the yaml file 
        #with open (yaml_path) as stream: yml_meta = yaml.safe_load(stream)
        #timestamp = datetime.strptime(yml_meta['extent']['center_dt'], '%Y-%m-%d %H:%M:%S')
        wofl = rxr.open_rasterio(wofl)
        logging.info(wofl)
        #wofl['time'] = timestamp
        wofls.append(wofl)     
        # Close the mask files
        wofl.close()

    logging.info('OUT OF THE LOOP')

    # Match res/projection and force align so the scenes can be concatenated
    wofls = [ wofls[i].rio.reproject_match(wofls[0]) for i in range(len(wofls)) ] 
    logging.info(f'REPROJECTED')
    
    # Avoid coord differences due to floats (without this the code crashes bc of misaligned coords)
    wofls = [wofls[i].assign_coords({
    "x": wofls[0].x,
    "y": wofls[0].y,}) for i in range(len(wofls))]

    logging.info(f'ASSIGNED COORDS')
    
    # # Align the scenes so they can be concatenated later
    wofls = [ xr.align(wofls[0], wofls[i], join="override", fill_value=0)[1] for i in range(len(wofls)) ] 
    logging.info(f'ALIGNED')

    # Concatenate xarrays into single dataset
    wofl_data = xr.concat(wofls, dim='time', fill_value=0).rename({'band': 'vals'})
    for i in wofls: i.close()
    logging.info(f'CONCATENATED: {wofl_data}')

    # Run timeseries analysis 
    out_data = perform_timeseries_analysis(wofl_data, 'vals', intermediate_product=None, no_data=0, operation="mean")
    logging.info(f'OUT DATA: {out_data}')

    # Write out_data to tif
    out_data.normalized_data.rio.to_raster(f'/home/spatialdays/Documents/testing-wofs/WOFLs/NormalizedData_WOFSUMMARY.tif')
    out_data.total_clean.rio.to_raster(f'/home/spatialdays/Documents/testing-wofs/WOFLs/TotalClean_WOFSUMMARY.tif')
    out_data.total_data.rio.to_raster(f'/home/spatialdays/Documents/testing-wofs/WOFLs/TotalData_WOFSUMMARY.tif')


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    # # Directory of directories 
    # big_dir = '/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/'
    # scene_dirs = glob.glob(f'{big_dir}LC08_L2SR_075072_*_tmp/')

    # scene_dir = f'{big_dir}LC08_L2SR_075072_20221011_tmp/'
    # scene_data = stack_bands(scene_dir)
    # logging.info(f'Finished stacking bands')

    # scene_name = scene_dir.split('/')[-2]
    # logging.info(f'SCENE NAME: {scene_name}')

    # # Define clear mask path
    # clear_mask = f'{big_dir}{scene_name}_masked/{scene_name}_clear_mask.tif'
    # clear_mask = clear_mask_to_boolean(clear_mask)

    # # Run wofs classification
    # wofs_data = wofs_classify(scene_data, clean_mask=clear_mask).astype('uint8')
    # logging.info(f'Finished wofs classification:')
    # wofs_data.wofs.rio.to_raster(f'/home/spatialdays/Documents/testing-wofs/WOFLs/{scene_name}_WOFL.tif')


    # Define folder that holds WOFLs
    wofl_dir = f'/home/spatialdays/Documents/testing-wofs/WOFLs/*.tif'

    # Run wofs summary
    wofs_summary(wofl_dir=wofl_dir)








    # Set the scene directory
    #scene_dir = '/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/LC08_L2SR_075072_20220605_tmp/'

    #scene_data = stack_bands(scene_dir)
    #logging.info(f'Finished stacking bands')

    # Define clear mask path 
    #clear_mask = '/home/spatialdays/Documents/testing-wofs/test_masking/Tile7572/LC08_L2SR_075072_20220605_tmp_masked/LC08_L2SR_075072_20220605_tmp_clear_mask.tif'
    #clear_mask = clear_mask_to_boolean(clear_mask)

    # Run wofs classification
    #wofs_data = wofs_classify(scene_data, clean_mask=clear_mask)

    #logging.info(f'Finished wofs classification: {wofs_data}')

    #wofs_data.wofs.rio.to_raster('WOFS_OUTPUT.tif')





