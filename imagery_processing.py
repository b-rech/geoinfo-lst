# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 21:29:33 2023

@author: brech
"""

# %% Initialization

# Required libraries
import os
import pandas as pd
import geopandas as gpd
import ee
import feather
import geemap

# ee.Authenticate()
ee.Initialize()

os.chdir('scripts')
import auxiliary_functions as aux
os.chdir('../')

# %% Data upload & imagery selection

# Weather station
station = ee.Geometry.Point([-46.61999999, -23.49638888])

# Area to be considered
aoi = station.buffer(distance=5000, proj='EPSG:31983')

# Landsat 8 Collection 2 Tier 1 TOA Reflectance
initial_coll = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
    .filterDate(start='2013-03-01', opt_end='2023-08-31')
    .filterBounds(aoi)
    .filterMetadata('IMAGE_QUALITY_OLI', 'equals', 9)
    )

# Load meteorological data
met_data = pd.read_csv('input_data\\dados_A701_H_2013-01-01_2023-08-31.csv',
                       skiprows=10, sep=';')

met_data = met_data.iloc[:, range(0, 6)]

# Rename columns
met_data.columns = ['date', 'hour', 'p_atm', 't_air', 'dew', 'rh']

# Format hour and convert relative humidity to proportion
met_data['hour'] = (met_data.loc[:, 'hour']/100).astype('int')
met_data['rh'] = met_data.loc[:, 'rh']/100

# Create a column of datetimes
met_data['datetime'] = pd.to_datetime(met_data.date)
met_data['datetime'] = [date.replace(hour=time) for date, time in
                        zip(met_data.datetime, met_data.hour)]

# Remove missing data and select only values between 8h and 17h
met_data = met_data[(met_data.hour >= 8) & (met_data.hour <= 17)]
met_data = met_data.dropna()

# %% Cloud cover assessment


def get_cloud_percent(image):
    """
    This function calculates the cloud cover proportion within the geometry
    and adds it to the metadata as a new attribute called CLOUDINESS.
    """

    # Select cloud band
    # It is assigned 1 to the selected pixels and 0 to the others
    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    cloud_band = (image.select(['QA_PIXEL']).bitwiseAnd(1 << 6).eq(0))

    # Generate cloud proportion at each feature and the max is selected
    # Since the values are just 0 e 1, the mean is equal to the proportion
    # An ee.Dictionary is generated with the key renamed to "CLOUDINESS"
    cloud_percent = (
        cloud_band.reduceRegion(**{
            'geometry': aoi,
            'reducer': ee.Reducer.mean(),
            'scale': 30}
            ).get('QA_PIXEL')
        )

    # Add information to image metadata
    return image.set({'CLOUDINESS': cloud_percent})


# Select only images with no cloud cover over the samples
selected_coll = (
    initial_coll.map(get_cloud_percent)
    .filterMetadata('CLOUDINESS', 'not_greater_than', 0.01)
    )


# %% LSE and LST retrieval

# Create ee.Dictionary with dew point temperature
dew_dict = ee.Dictionary.fromLists(
    keys=met_data.datetime.astype(str).tolist(),
    values=met_data.dew.tolist())


# Function to set meteorological data to the scenes
def get_met_data(image):

    minutes = image.date().get('minute')
    seconds = image.date().get('second')
    hour = ee.Algorithms.If(minutes.gte(30), 1, 0)

    # Remove minutes and seconds
    # If minutes >= 30, round to the next hour
    datetime = (image.date()
                .advance(minutes.multiply(-1), 'minute')
                .advance(seconds.multiply(-1), 'second')
                .advance(hour, 'hour')
                ).format('YYYY-MM-dd HH:mm:ss')

    # Get dew point
    dew = dew_dict.getNumber(datetime)

    # Get real vapor pressure and convert from hPa to g/cm²
    # (Atmospheric Water Vapor - AWV)
    awv_expression = '0.098 * 6.1078 * 10**(7.5 * dew / (237.3 + dew))'

    awv = ee.Number.expression(expression=awv_expression, vars={'dew': dew})

    return image.set({'AWV': awv})


# Processing
lse_coll = (selected_coll
            .map(aux.lse)  # Get LSE of each scene
            .map(get_met_data)  # get meteorological variables
            .map(aux.brightness_to_rad)  # Convert to radiance
            )

# Mean and median LSE
mean_lse = lse_coll.select('LSE').mean().rename('LSE_mean')
median_lse = lse_coll.select('LSE').median().rename('LSE_median')

# Get LST
final_coll = (lse_coll
              .map(lambda image: image.addBands([mean_lse, median_lse]))
              .map(aux.lst)
              .map(aux.lst_mean_lse)
              .map(aux.lst_median_lse)
              )


# %% Extract data


# Define function to be mapped over the collection
def sample_images(img):

    sampled = (
        img.select(['LSE', 'LST', 'LST_mean', 'LST_median'])
        .sample(region=aoi, scale=30, numPixels=1000, seed=999)
        .map(lambda ft: ft.set({'date': img.date().format()}))
        )

    # Return ee.FeatureCollection
    return sampled


# Sample data
sampled_coll = final_coll.map(sample_images).flatten()

# Extract to a Pandas data frame
final_dataframe = pd.DataFrame({
    'datetime': sampled_coll.aggregate_array('date').getInfo(),
    'lse': sampled_coll.aggregate_array('LSE').getInfo(),
    'lst': sampled_coll.aggregate_array('LST').getInfo(),
    'lst_mea': sampled_coll.aggregate_array('LST_mean').getInfo(),
    'lst_med': sampled_coll.aggregate_array('LST_median').getInfo()
    })

final_dataframe['datetime'] = pd.to_datetime(final_dataframe.datetime)

# Save to feather
feather.write_dataframe(final_dataframe, 'generated_data\\lst_dataset.feather')


# %% Generate images


# Function for calculating difference between LST estimates
def get_abs_diff(image):

    # Calculate absolute difference with LST_mean
    diff1 = (image
             .select('LST')
             .subtract(image.select('LST_mean'))
             .abs()
             .rename('diff_mean'))

    # Calculate absolute difference with LST_median
    diff2 = (image
             .select('LST')
             .subtract(image.select('LST_median'))
             .abs()
             .rename('diff_median'))

    return image.addBands([diff1, diff2])


# Map function
final_coll = final_coll.map(get_abs_diff)

# Get mean absolute error (difference) spatialized
mae_mean = final_coll.select('diff_mean').mean()
mae_median = final_coll.select('diff_median').mean()

# Download images

# Mean image
geemap.ee_export_image(final_coll.select(['B4', 'B3', 'B2']).median(),
                       filename='generated_data\\img_median.tif',
                       scale=30,
                       region=station.buffer(
                           distance=10000, proj='EPSG:31983'),
                       file_per_band=True)

# MAE of LST with mean LSE
geemap.ee_export_image(mae_mean,
                       filename='generated_data\\img_mae_mean.tif',
                       scale=30,
                       region=station.buffer(
                           distance=10000, proj='EPSG:31983'))

# MAE of LST with median LSE
geemap.ee_export_image(mae_median,
                       filename='generated_data\\img_mae_median.tif',
                       scale=30,
                       region=station.buffer(
                           distance=10000, proj='EPSG:31983'))

# Download weather station buffer
geemap.ee_export_vector(ee.FeatureCollection(aoi),
                        'generated_data\\station_buffer.kmz')
