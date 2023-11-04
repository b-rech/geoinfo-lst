# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:13:14 2023

@author: brech
"""

# %% Initialization


# Required libraries
import ee
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns

# GEE initialization
ee.Initialize()


# %% Cloud Mask


def cloud_mask(image):
    '''
    Select Bit 6 (clouds and dilated clouds) from QA_PIXEL band and remove
    cloudy pixels from the image.
    '''

    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    clear = image.select('QA_PIXEL').bitwiseAnd(1 << 6)

    return image.updateMask(clear)


# %% Brightness temperature to radiance


def brightness_to_rad(image):
    '''
    This function takes B10 from USGS Landsat 8 Collection 2 Tier 1
    TOA Reflectance (brightness temperature) and converts to radiance.
    '''

    k1 = image.getNumber('K1_CONSTANT_BAND_10')
    k2 = image.getNumber('K2_CONSTANT_BAND_10')

    image = image.addBands(
        image.expression(expression='k1 / (exp(k2 / B10) - 1)',
                         opt_map={'k1': k1, 'k2': k2,
                                  'B10': image.select('B10')}).rename('B10R')
        )

    return image


# %% Vector to ee.FeatureCollection


def vect_to_fc(vector):
    '''
    Transforms a multipolygon vector (geopandas Dataframe) into
    an ee.FeatureCollection.
    The vector must have a field called "surface".
    '''

    # Empty list to save the polygons
    polygons = ee.List([])

    # Iterate over each polygon
    for pol in range(0, len(vector)):

        # Select the feature of interest
        polygon = (
            np.dstack(vector.geometry[pol].exterior.coords.xy).tolist()
            )

        # Create a ee.Feature with the external coordinates
        geometry = ee.Feature(ee.Geometry.Polygon(polygon))

        # Set an attribute from the vector to the geometry
        geometry = geometry.set({'surface': vector.surface[pol]})

        # Append the geometry to the list
        polygons = polygons.add(geometry)

    return ee.FeatureCollection(polygons)


# %% Land Surface Emissivity


def lse(image):
    '''
    This function retrieves Land Surface Emissivity from Landsat 8 TOA bands
    using the formulation from Li and Jiang (2018).
    '''

    # Calculate Normalized Difference Vegetation Index (NDVI)
    image = image.addBands(
        image.normalizedDifference(['B5', 'B4'])
        .rename('NDVI')
        )

    # Constants adopted
    NDVIs = 0.2  # Soil NDVI
    NDVIv = 0.5  # Vegetation NDVI
    lses = 0.971  # Soil NDVI
    lsev = 0.982  # Vegetation NDVI
    gf = 0.55  # Geometrical Factor

    # Fractional Vegetation Cover (FVC)
    fvc_expression = '''
        (NDVI <= 0.2) ? 0 :
        (NDVI > 0.2 && NDVI < 0.5) ? ((NDVI - NDVIs) / (NDVIv - NDVIs))**2 :
        1
        '''

    image = image.addBands(
        image.expression(expression=fvc_expression,
                         opt_map={'NDVI': image.select('NDVI'),
                                  'NDVIs': NDVIs, 'NDVIv': NDVIv})
        .rename('FVC')
        )

    # Cavity Effect (CE)
    ce_expression = '''
    (NDVI < 0.5) ? ((1 - lses) * lsev * gf * (1 - FVC)) : 0.005
    '''

    image = image.addBands(
        image.expression(expression=ce_expression,
                         opt_map={'FVC': image.select('FVC'),
                                  'NDVI': image.select('NDVI'),
                                  'lses': lses, 'lsev': lsev, 'gf': gf}
                         ).rename('CE')
        )

    # Land Surface Emissivity (LSE)
    lse_expression = '''
        (NDVI < 0.2) ? (0.98 - 0.14 * B2 + 0.17 * B3 - 0.036 * B4
                        - 0.083 * B5 + 0.158 * B6 - 0.149 * B7) :
        (NDVI >= 0.2 && NDVI <= 0.5) ? (lsev * FVC  + lses *
                                        (1 - FVC) + CE) :
        (lsev + CE)
         '''

    image = image.addBands(
        image.expression(
            expression=lse_expression,
            opt_map={'NDVI': image.select('NDVI'),
                     'FVC': image.select('FVC'),
                     'CE': image.select('CE'),
                     'B2': image.select('B2'),
                     'B3': image.select('B3'),
                     'B4': image.select('B4'),
                     'B5': image.select('B5'),
                     'B6': image.select('B6'),
                     'B7': image.select('B7'),
                     'lses': lses, 'lsev': lsev}
            ).rename('LSE')
        )

    return image


# %% Land Surface Temperature


def lst(image):
    '''
    This function retrieves Land Surface Temperature from Landsat 8 TOA bands
    using the formulation from Wang et al. (2019).
    '''

    # Coefficients for B(T) model

    # # AWV in [0, 2]
    # a = [-0.28009, 1.257429, 0.275109, -1.32876,
    #      -0.1696, 0.999069, 0.033453, 0.015232]

    # # AWV in [2, 4]
    # b = [-0.60336, 1.613485, -4.98989, 2.772703,
    #      -1.04271, 1.739598, -0.54978, 0.129006]

    # # AWV in [4, 7]
    # c = [2.280539, 0.918191, -38.3363, 13.82581,
    #      -1.75455, 5.003919, -1.62832, 0.196687]

    # # Full range
    # d = [-0.4107, 1.493577, 0.278271, -1.22502,
    #      -0.31067, 1.022016, -0.01969, 0.036001]

    # Calculate blackbody radiance
    bbr_expression = '''
        (AWV <= 2) ?
        (- 0.28009 + 1.257429 * AWV
         + (0.275109 - 1.32876 * AWV - 0.1696 * (AWV**2)) / LSE
         + (0.999069 + 0.033453 * AWV + 0.015232 * (AWV**2)) * B10/LSE) :

        (AWV > 2 && AWV <= 4) ?
        (- 0.60336 + 1.613485 * AWV
         + (- 4.98989 + 2.772703 * AWV - 1.04271 * (AWV**2)) / LSE
         + (1.739598 - 0.54978 * AWV + 0.129006 * (AWV**2)) * B10/LSE) :

        (AWV > 4 && AWV <= 7) ?
        (2.280539 + 0.918191 * AWV
         + (- 38.3363 + 13.82581 * AWV - 1.75455 * (AWV**2)) / LSE
         + (5.003919 - 1.62832 * AWV + 0.196687 * (AWV**2)) * B10/LSE) :

        (- 0.4107 + 1.493577 * AWV
         + (0.278271 - 1.22502 * AWV - 0.31067 * (AWV**2)) / LSE
         + (1.022016 - 0.01969 * AWV + 0.036001 * (AWV**2)) * B10/LSE)
        '''

    image = image.addBands(
        image.expression(
            expression=bbr_expression,
            opt_map={'AWV': image.getNumber('AWV'),
                     'LSE': image.select('LSE'),
                     'B10': image.select('B10R')}
            ).rename('BBR')
        )

    # Retrieve LST
    lst_expression = '(c2 / lambda) / (log(c1 / (lambda**5 * BBR) + 1))'

    image = image.addBands(
        image.expression(
            expression=lst_expression,
            opt_map={'c1': 1.19104E+08,
                     'c2': 1.43877E+04,
                     'lambda': 10.904,
                     'BBR': image.select('BBR')}
            ).rename('LST')
        )

    return image


# %% Land Surface Temperature with mean LSE


def lst_mean_lse(image):
    '''
    This function retrieves Land Surface Temperature from Landsat 8 TOA bands
    using the formulation from Wang et al. (2019) and mean LSE.
    '''

    # Coefficients for B(T) model

    # # AWV in [0, 2]
    # a = [-0.28009, 1.257429, 0.275109, -1.32876,
    #      -0.1696, 0.999069, 0.033453, 0.015232]

    # # AWV in [2, 4]
    # b = [-0.60336, 1.613485, -4.98989, 2.772703,
    #      -1.04271, 1.739598, -0.54978, 0.129006]

    # # AWV in [4, 7]
    # c = [2.280539, 0.918191, -38.3363, 13.82581,
    #      -1.75455, 5.003919, -1.62832, 0.196687]

    # # Full range
    # d = [-0.4107, 1.493577, 0.278271, -1.22502,
    #      -0.31067, 1.022016, -0.01969, 0.036001]

    # Calculate blackbody radiance
    bbr_expression = '''
        (AWV <= 2) ?
        (- 0.28009 + 1.257429 * AWV
         + (0.275109 - 1.32876 * AWV - 0.1696 * (AWV**2)) / LSE
         + (0.999069 + 0.033453 * AWV + 0.015232 * (AWV**2)) * B10/LSE) :

        (AWV > 2 && AWV <= 4) ?
        (- 0.60336 + 1.613485 * AWV
         + (- 4.98989 + 2.772703 * AWV - 1.04271 * (AWV**2)) / LSE
         + (1.739598 - 0.54978 * AWV + 0.129006 * (AWV**2)) * B10/LSE) :

        (AWV > 4 && AWV <= 7) ?
        (2.280539 + 0.918191 * AWV
         + (- 38.3363 + 13.82581 * AWV - 1.75455 * (AWV**2)) / LSE
         + (5.003919 - 1.62832 * AWV + 0.196687 * (AWV**2)) * B10/LSE) :

        (- 0.4107 + 1.493577 * AWV
         + (0.278271 - 1.22502 * AWV - 0.31067 * (AWV**2)) / LSE
         + (1.022016 - 0.01969 * AWV + 0.036001 * (AWV**2)) * B10/LSE)
        '''

    image = image.addBands(
        image.expression(
            expression=bbr_expression,
            opt_map={'AWV': image.getNumber('AWV'),
                     'LSE': image.select('LSE_mean'),
                     'B10': image.select('B10R')}
            ).rename('BBR_mean')
        )

    # Retrieve LST
    lst_expression = '(c2 / lambda) / (log(c1 / (lambda**5 * BBR) + 1))'

    image = image.addBands(
        image.expression(
            expression=lst_expression,
            opt_map={'c1': 1.19104E+08,
                     'c2': 1.43877E+04,
                     'lambda': 10.904,
                     'BBR': image.select('BBR_mean')}
            ).rename('LST_mean')
        )

    return image


# %% Land Surface Temperature with median LSE


def lst_median_lse(image):
    '''
    This function retrieves Land Surface Temperature from Landsat 8 TOA bands
    using the formulation from Wang et al. (2019) and median LSE.
    '''

    # Coefficients for B(T) model

    # # AWV in [0, 2]
    # a = [-0.28009, 1.257429, 0.275109, -1.32876,
    #      -0.1696, 0.999069, 0.033453, 0.015232]

    # # AWV in [2, 4]
    # b = [-0.60336, 1.613485, -4.98989, 2.772703,
    #      -1.04271, 1.739598, -0.54978, 0.129006]

    # # AWV in [4, 7]
    # c = [2.280539, 0.918191, -38.3363, 13.82581,
    #      -1.75455, 5.003919, -1.62832, 0.196687]

    # # Full range
    # d = [-0.4107, 1.493577, 0.278271, -1.22502,
    #      -0.31067, 1.022016, -0.01969, 0.036001]

    # Calculate blackbody radiance
    bbr_expression = '''
        (AWV <= 2) ?
        (- 0.28009 + 1.257429 * AWV
         + (0.275109 - 1.32876 * AWV - 0.1696 * (AWV**2)) / LSE
         + (0.999069 + 0.033453 * AWV + 0.015232 * (AWV**2)) * B10/LSE) :

        (AWV > 2 && AWV <= 4) ?
        (- 0.60336 + 1.613485 * AWV
         + (- 4.98989 + 2.772703 * AWV - 1.04271 * (AWV**2)) / LSE
         + (1.739598 - 0.54978 * AWV + 0.129006 * (AWV**2)) * B10/LSE) :

        (AWV > 4 && AWV <= 7) ?
        (2.280539 + 0.918191 * AWV
         + (- 38.3363 + 13.82581 * AWV - 1.75455 * (AWV**2)) / LSE
         + (5.003919 - 1.62832 * AWV + 0.196687 * (AWV**2)) * B10/LSE) :

        (- 0.4107 + 1.493577 * AWV
         + (0.278271 - 1.22502 * AWV - 0.31067 * (AWV**2)) / LSE
         + (1.022016 - 0.01969 * AWV + 0.036001 * (AWV**2)) * B10/LSE)
        '''

    image = image.addBands(
        image.expression(
            expression=bbr_expression,
            opt_map={'AWV': image.getNumber('AWV'),
                     'LSE': image.select('LSE_median'),
                     'B10': image.select('B10R')}
            ).rename('BBR_median')
        )

    # Retrieve LST
    lst_expression = '(c2 / lambda) / (log(c1 / (lambda**5 * BBR) + 1))'

    image = image.addBands(
        image.expression(
            expression=lst_expression,
            opt_map={'c1': 1.19104E+08,
                     'c2': 1.43877E+04,
                     'lambda': 10.904,
                     'BBR': image.select('BBR_median')}
            ).rename('LST_median')
        )

    return image
