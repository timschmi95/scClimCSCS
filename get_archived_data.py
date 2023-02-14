import datetime
import os
import re
from typing import Tuple
from zipfile import ZipFile
import numpy as np
import radlib as rad
import xarray as xr
from PIL import Image
import warnings

# username = "tschmid" #os.getlogin()
username = "tschmid"
if os.environ.get("METRANETLIB_PATH") is None:
    os.environ["METRANETLIB_PATH"] = f"/users/{username}/metranet/"


BASE_DIR_RADAR = (
    "/store/msrad/radar/swiss/data/"  # on MCH servers: "/repos/repos_prod/radar/swiss"
)
TEMP_ZIP_DIR = f"/scratch/{username}/zip_temp_dir"
TEMP_GRID_DIR = f"/scratch/{username}/grid_temp_dir"

# TODO: Properly solve the situation with the necessary files within the repository
MCH_NC_EXAMPLE = f"/users/{username}/data/config_files/MZC_example.nc"
CPC_LUT_PATH = "/users/maregger/PhD/lut/"


RADAR_PRODUCTS = {  # values are filename endings of products
    "dBZC": "2400.+.845",   #daily POH
    "dLZC": "2400.+.801", 
    "dMZC": "2400.+.850",   #daily MESHS
    "dCZC": "2400.+.801",
    "BZC": ".+.845",        #POH
    "EZC15": ".+.815",
    "EZC45": ".+.845",
    "MZC": ".+.850",        #MESHS
    "LZC": ".+.801",
    "RZC": ".+.801",
    "CZC": ".+.801",
    "HZT": ".+.800",
    "TRTC": ".+.trt",
    "YM": ".+.8",
    "ML": ".+.0",
    "CPCH_5": ".+_00005.801.gif",
    "CPCH_60": ".+_00060.801.gif",
}


projection_dict = {
    "proj": "somerc",
    "lon_0": 7.43958333333333,
    "lat_0": 46.9524055555556,
    "k_0": 1,
    "x_0": 600000,
    "y_0": 200000,
    "ellps": "bessel",
    "towgs84": "674.374,15.056,405.346,0,0,0,0",
    "units": "m",
    "no_defs": "",
}


def read_cpc_file(filepath: str, lut: str = "medium") -> np.ndarray:
    """Reads the given CPC gif file and converts the 8-bit values into mm/h.
    The conversion is done using Look-up-tables which are currently stored in
    three seperate .npy files. The LUTs were generated from the values given on
    the CPC confluence page
    (https://service.meteoswiss.ch/confluence/display/CRS/CombiPrecip). 
    They are valid for CPC version 3.5 and newer. For CPC there is always a
    minimum, medium, and maximum estimation available. The respective LUT
    can be chosen in the "lut argument

    Args:
        filepath (str): Path of the CPC gif file which shall be read.
        lut (str, optional): Possibility to select either the minimum,
        medium or maximum value for CPC. Defaults to "medium".

    Raises:
        ValueError: This error is raised if an invalid lut name is given

    Returns:
        ndarray: array containing CPC values in mm/h
    """
    image = Image.open(filepath)
    if lut == "medium":
        lut = np.load("/users/maregger/PhD/lut/cpc_lut_minimum.npy")
    elif lut == "minimum":
        lut = np.load("/users/maregger/PhD/lut/cpc_lut_minimum.npy")
    elif lut == "maximum":
        lut = np.load("/users/maregger/PhD/lut/cpc_lut_maximum.npy")
    else:
        raise ValueError("Invalid CPC Lut selected")

    image = lut[image]

    return image


def get_netcdf(varname, date):
    """gets netcdf file for specific hailday (6UTC to 6UTC)

    Args:
        varname (str): 3-letter description of metranet radar_variable
        date (datetime.datetime): date 

    Returns:
        ds_out (xr.Dataset): xarray dataset daily maximum values
        of the chosen variable on the given hailday (6 to 6 UTC)
    """
    date_p_1 = date + datetime.timedelta(days=1)
    tstamp1 = date.strftime("%Y%m%d") + "060000"
    tstamp2 = date_p_1.strftime("%Y%m%d") + "060000"

    npy_arr = get_combined_max_radar_grid(varname, tstamp1, tstamp2)
    ds_out = npy_to_netcdf(npy_arr, varname, date)
    return ds_out


def npy_to_netcdf(np_arr, varname, date_dt=None):
    """convert np.ndarrays to netCDF datasets, based on the example
    file MCH_NC_EXAMPLE. The extent must be the MCH radardomain
    (710(chx) x 640(chy) gridpoints).
    If input is a directory, it automatically converts all .npy files
    from that dictionary to one netcdf.

    Args:
        np_arr (path,directory, or np.ndarray): 
            path: path to .npy file
            directory: dir which contains .npy files, named as follows:
                {varname}_%Y%m%d%H%M%S.npy (e.g. MZC_20210630060000.npy)
            np.ndarray: numpy array 
            ..containing a radarvariable with 710x640 gridpoints
        varname (str): 
            MCH metranet radarname (e.g. MZC for MESHS)
        date_dt (dt.datetime, optional): 
            Date (or time) of the observation in 'np_arr'. Defaults to None.

    Returns:
        ds_out (xr.Dataset): xarray dataset of the numpy array
    """
    # load example netcdf file with correct coords
    nc_filepath = str(MCH_NC_EXAMPLE)
    ncfile = xr.open_dataset(nc_filepath)

    if isinstance(np_arr, np.ndarray):  # if input is a ndarray object
        arr = np.flip(np_arr, axis=[0])

        # convert MESHS from cm to mm
        if varname in ["MZC", "meshs"]:
            arr = arr * 10

        ds_out = xr.Dataset(
            {varname: (("chy", "chx"), arr)}, coords=ncfile.coords
        ).isel(time=0)
        # Override time dimension with actual timestamp
        ds_out["time"] = date_dt

    elif np_arr.endswith(".npy"):  # if input is a .npy file
        arr = np.flip(np.load(np_arr), axis=[0])
        # convert MESHS from cm to mm
        if varname in ["MZC", "meshs"]:
            arr = arr * 10

        ds_out = xr.Dataset(
            {varname: (("chy", "chx"), arr)}, coords=ncfile.coords
        ).drop("time")
    elif os.path.isdir(np_arr):  # if input is directory of .npy files
        np_arrs = os.listdir(np_arr)
        for file in np_arrs:
            if file.endswith(".npy"):
                timestamp = file.replace("%s_" % varname, "").replace(".npy", "")
                arr = np.flip(np.load(os.path.join(np_arr, file)), axis=[0])

                # convert MESHS from cm to mm
                if varname in ["MZC", "meshs"]:
                    arr = arr * 10
                # arr = np.load(npz)
                ds = xr.Dataset(
                    {varname: (("chy", "chx"), arr)}, coords=ncfile.coords
                ).isel(time=0)
                ds["time"] = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                if file == np_arrs[0]:
                    ds_out = ds
                else:
                    ds_out = xr.concat([ds_out, ds], dim="time")
    else:
        TypeError("np_arr is neither .npy file nor directory nor np.ndarray object")

    return ds_out


def unzip_radar_files(
    zipfile_path: str, unzipped_file_path: str, filename_pattern: str
) -> str:
    """This method unzips the selected files from the given paths. If the
    files are not found an error is raised

    Parameters
    ----------
    zipfile_path : str
        The path to the location of the zipfile
    unzipped_file_path : str
        The path to the location where the file will be unzipped to
    filename_pattern : str
        A regex pattern of the file which is searched for in the zipfile

    Returns
    -------
    str
        The path to the unziped radar file

    Raises
    ------
    FileNotFoundError
        This error is raised if the requested file does not exist
    """
    # Unzipp file only if it isn't already
    file_exists = False
    try:
        for filename in os.listdir(unzipped_file_path):
            if filename_pattern.match(filename):
                file_exists = True
                file_path_out = unzipped_file_path + "/" + filename
                break
    except FileNotFoundError:
        pass
    try:
        if not file_exists:
            with ZipFile(zipfile_path, "r",) as zip_object:
                list_of_filenames = zip_object.namelist()
                for filename in list_of_filenames:
                    if filename_pattern.match(filename):
                        zip_object.extract(filename, unzipped_file_path)
                        file_path_out = unzipped_file_path + "/" + filename
                        file_exists = True
                        break
    except FileNotFoundError:
        pass

    if not file_exists:
        raise FileNotFoundError("Requested zip file does not exist.")

    return file_path_out


def build_zip_file_paths(
    timestamp: str, product: str, radar_elevation: str = 1, radar="A"
) -> Tuple[str, str, str]:
    """Returns the paths which are necessary to unzip the selected product

    Parameters
    ----------
    timestamp : str
        timestamp of requested file in the format "20190701233000"
    product : str
        One of the supported product types e.g. 'dBZC'
    radar_elevation : str, optional
        selected beam elevation with 1 beeing the lowest beam (-0.2°), by
        default 1, Necessary for HYD, by default 1
    radar : str, optional
        for polar products the radar has to be selected options are "A", "D", "L", "P", "M"

    Returns
    -------
    tuple[str,str,str]
        The zipfile_path, unzipped_file_path and a filename_pattern.

    Raises
    ------
    ValueError
        If an unsupported product is selected.

    """
    timestamp_obj = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    day_of_year = timestamp_obj.strftime("%-j")

    # getting path to unzipped file folder
    unzipped_file_path = (
        f"{TEMP_ZIP_DIR}/"
        f"{timestamp_obj.year}/"
        f"{timestamp_obj.month}/"
        f"{day_of_year}"
    )

    # Checking input
    if product not in RADAR_PRODUCTS:
        raise ValueError("Unsupported Product")

    # Build the zipfile path string
    ydoy_string = timestamp_obj.strftime("%y%j")

    if product in ["YM", "ML"]:
        product = product + radar

    basic_product_name = product
    if product in ["EZC15", "EZC45"]:
        basic_product_name = "EZC"

    if product in ["CPCH_5", "CPCH_60"]:
        basic_product_name = "CPCH"

    # # BZC Analysis Version adds the H
    if product in ["BZC", "MZC"]:
        basic_product_name = product +"H"

    file_name = basic_product_name + ydoy_string
    file_name += ".zip"

    zipfile_path = BASE_DIR_RADAR + "/"
    zipfile_path += str(timestamp_obj.year) + "/"
    zipfile_path += ydoy_string + "/"
    zipfile_path += file_name
    # print(zipfile_path)
    # Select the correct file ending
    if product in ("dBZC", "dMZC", "dCZC", "dLZC", "dEZC"):
        unzipped_file_name = product[1:] + ydoy_string

    else:
        # HZT & CPCH_60 is only available for every hour
        if product in ["HZT", "CPCH_60"]:
            timestamp_obj = timestamp_obj.replace(minute=0)
        # TRT has different naming scheme
        if product == "TRTC":
            unzipped_file_name = "CZC"
        elif product in ["BZC","MZC"]:
            unzipped_file_name = product
        elif product in ["CPCH_5", "CPCH_60"]:
            unzipped_file_name = "CPC"
        else:
            unzipped_file_name = basic_product_name
        unzipped_file_name += timestamp_obj.strftime("%y%j%H%M")

    if product[0:2] in ["YM", "ML"]:
        product = product[0:2]

    unzipped_file_name += RADAR_PRODUCTS[product]

    if product[0:2] in ["YM", "ML"]:
        unzipped_file_name += str(radar_elevation).zfill(2)

    # add wildcard forfile path (because in 2008 all files
    # are stored in a subfolder for CZC)
    unzipped_file_name = ".*" + unzipped_file_name

    filename_pattern = re.compile(unzipped_file_name)
    return zipfile_path, unzipped_file_path, filename_pattern


def prepare_gridded_radar_data_from_zip(
    product: str, timestamp: str, reader: str = "C"
) -> np.ndarray:
    """Returns the numpy ndarray of a gridded radar product for a given date
    and time. Currently supports 'dBZC','dLZC','dMZC','dCZC','BZC','EZC','MZC'
    'LZC','RZC','CZC','HZT', 'CPCH_5','CPCH_60' but more products can easily be added.
    CPCH_60 & HZT will default to the minute 0 of the hour. 
    An input of 15.55 will return the value for the hour 15:00
    Product description is available in Confluence
    (https://service.meteoswiss.ch/confluence/display/CRS/Operational+Radar).

    The ndarray will be unzipped from the archive file and stored in the
    TEMP_ZIP_DIR set in the config file. If it was already unzipped
    before it will just load the existing file.


    Parameters
    ----------
    product : str
        One of the supported product types e.g. 'dBZC'
    timestamp : str
        timestamp of requested file in the format "20190701233000"
    reader : str, optional


    Returns
    -------
    ndarray
        A ndarray containing the radar data grid.

    Raises
    ------
    ValueError
        If the timestamp has an invalid format.
    FileNotFoundError
        If there is no file available for the requested parameter combination.

    """
    # creating timestamp from input
    if isinstance(timestamp, str) is False:
        raise ValueError("Timestamp is not valid!")
    if len(timestamp) != 14:
        raise ValueError("Timestamp is not valid!")

    zipfile_path, unzipped_file_path, filename_pattern = build_zip_file_paths(
        timestamp, product
    )

    file_path_out = unzip_radar_files(
        zipfile_path, unzipped_file_path, filename_pattern
    )

    # read the numpy ndarray from the file
    if not os.path.exists(file_path_out):
        print("File doesnt exist: ")
        print(file_path_out)

    if product in ["CPCH_5", "CPCH_60"]:
        # CPC files are in gif format, they need a different reading routine
        values = read_cpc_file(filepath=file_path_out)
    else:
        if reader == "C":
            values = rad.read_file(file=file_path_out, physic_value=True).data
        if reader == "Python":
            raise NotImplementedError(
                "The Python version of the metranet reader \
                is not yet implemented."
            )
            """  radar_object = read_cartesian_metranet(
                filename=file_path_out, reader="python"
            )
            print(radar_object.fields.keys())
            # TO-DO: not yet working correctly needs to be adapted
            values = np.squeeze(radar_object.fields["probability_of_hail"]["data"])
            values = np.flipud(values)"""
        # TODO: add this part to martins code too
    # Remove files from temp_dir
    os.remove(file_path_out)
    return values


def get_combined_max_radar_grid(
    product: str, timestamp1: str, timestamp2: str
) -> np.ndarray:

    if product in ["MZC","BZC"]:
        warnings.warn(f"by default {product}H will be used, which uses the 0C-line from the Analysis")

    start_date = datetime.datetime.strptime(timestamp1, "%Y%m%d%H%M%S")
    end_date = datetime.datetime.strptime(timestamp2, "%Y%m%d%H%M%S")
    temp_date = start_date - datetime.timedelta(
        minutes=5
    )  # ensure starting at first timestep
    skipped_timesteps = 0
    while temp_date < end_date:
        temp_date += datetime.timedelta(minutes=5)
        try:
            if "grid" in locals():
                grid = np.maximum(
                    grid,
                    prepare_gridded_radar_data_from_zip(
                        product=product,
                        timestamp=datetime.datetime.strftime(temp_date, "%Y%m%d%H%M%S"),
                    ),
                )
            else:
                grid = prepare_gridded_radar_data_from_zip(
                    product=product,
                    timestamp=datetime.datetime.strftime(temp_date, "%Y%m%d%H%M%S"),
                )

        except (AttributeError, FileNotFoundError) as err:
            # AttributeError: 'NoneType' object has no attribute 'data' ():
            # in prepare_gridded_radar_data_from_zip ->rad.read_file --> file cannot be read
            # FileNotFoundError: File is not in archive
            warnings.warn(
                f"{temp_date} is not in archive (or not readable). This timestep is skipped"
            )
            skipped_timesteps += 1
            # If >10 days are missing add a nan-only slice for this day and print a warning
            if skipped_timesteps > 10:
                grid = np.full((640, 710), np.nan)
                warnings.warn(
                    f"{temp_date}: number of skipped steps is >10. return empty array for this date"
                )
                break
    return grid


def save_multiple_radar_grids(
    product: str, timestamp1: str, timestamp2: str
) -> np.ndarray:

    if product in ["MZC","BZC"]:
        warnings.warn(f"by default {product}H will be used, which uses the 0C-line from the Analysis")

    start_date = datetime.datetime.strptime(timestamp1, "%Y%m%d%H%M%S")
    end_date = datetime.datetime.strptime(timestamp2, "%Y%m%d%H%M%S")
    grid = prepare_gridded_radar_data_from_zip(product=product, timestamp=timestamp1)
    temp_date = start_date
    while temp_date < end_date:
        temp_date += datetime.timedelta(minutes=5)
        temp_timestamp = datetime.datetime.strftime(temp_date, "%Y%m%d%H%M%S")
        grid = prepare_gridded_radar_data_from_zip(
            product=product, timestamp=temp_timestamp
        )
        np.save(
            "/scratch/%s/data/subdaily_npy/%s_%s.npy"
            % (username, product, temp_timestamp),
            grid,
        )


def get_cpc_quality_code(product: str, timestamp: str) -> int:
    """Returns the quality code which is stored in the filnames of the 
    CPCH_5 and CPCH_60 files


    Parameters
    ----------
    product : str
        One of the supported product types e.g. 'dBZC'
    timestamp : str
        timestamp of requested file in the format "20190701233000"
    reader : str, optional


    Returns
    -------
    ndarray
        A ndarray containing the radar data grid.

    Raises
    ------
    ValueError
        If the timestamp has an invalid format.
    FileNotFoundError
        If there is no file available for the requested parameter combination.

    """
    # creating timestamp from input
    if isinstance(timestamp, str) is False:
        raise ValueError("Timestamp is not valid!")
    if len(timestamp) != 14:
        raise ValueError("Timestamp is not valid!")
    if product not in ["CPCH_5", "CPCH_60"]:
        raise ValueError("Unsupported Product")

    zipfile_path, unzipped_file_path, filename_pattern = build_zip_file_paths(
        timestamp, product
    )

    file_path_out = unzip_radar_files(
        zipfile_path, unzipped_file_path, filename_pattern
    )

    # read the numpy ndarray from the file
    if not os.path.exists(file_path_out):
        print("File doesnt exist: ")
        print(file_path_out)

    quality_code_cpc = int(file_path_out.split("/")[-1].split("_")[0][-1])

    os.remove(file_path_out)
    return quality_code_cpc
