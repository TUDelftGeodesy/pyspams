import math

import numpy as np
import pandas as pd


def spams_model(xP, xE, xI, tau, meteo_df):
    """Compute SPAMS model using SPAMS parameters and meteo data
    from The Royal Netherlands Meteorological Institute (KNMI).

    Reference: Conroy et. al.. 2023. SPAMS: A new empirical model for
    soft soil surface displacement based on meteorological input data
    (https://doi.org/10.1016/j.geoderma.2023.116699).


    Parameters
    ----------
    xP : float
        Scaling factor for precipitation (mm/mm).
    xE : float
        Scaling factor for evapotranspiration (mm/mm).
    xI : float
        Irreversible constant, active during dry period (mm/day).
    tau : int
        Time constant (days).
    meteo_df : pd.DataFrame
        Daily precipitation and evapotranspiration (mm).

    Returns
    -------
    float
        Relative height, reversible, and irreversible in mm.
    """
    ## Get daily precipitation and evapotranspiration
    p = np.lib.stride_tricks.sliding_window_view(meteo_df["precip"], tau)
    e = np.lib.stride_tricks.sliding_window_view(meteo_df["evapo"], tau)

    ## Reversible
    reversible = np.sum(xP * p - xE * e, axis=1)

    ## Irreversible
    dry_flag = np.zeros(reversible.size)
    dry_flag[reversible < 0] = 1
    irreversible = np.cumsum(xI * dry_flag)

    ## Relative height in m
    height = reversible + irreversible

    return reversible, irreversible, height


def irreversible_rate(irreversible, xI, var_xI):
    """Irreversible rate in mm/year

    Parameters
    ----------
    irreversible : np.array
        SPAMS time series model for irreversible part.
    xI : float
        Irreversible constant, active during dry period (mm/day).
    var_xI : float
        Variance of the irreversible constant.

    Returns
    -------
    float
        Mean irreversible rate per year and its variance.
    """
    ## Dry flag: integer boolean masking the dry periods with 1
    dry_flag = np.insert(np.diff(irreversible / xI), 0, 0)

    ## Linear irreversible rate per year and its standard deviation
    vI = irreversible[-1] / (irreversible.shape[0] / 365.25)
    std_vI = np.sqrt(var_xI) * (np.sum(dry_flag) / (irreversible.shape[0] / 365.25))

    return vI, std_vI


def f_value(rss, dof):
    """F-value statistic: the weighted sum of squared residuals between InSAR and SPAMS,
    normalized by the degrees of freedom for each parcel.

    This value evaluates model suitability and identify whether an error is present in
    the mathematical model. Values close to one suggest model adequacy, while values
    larger than one indicate either model imperfections or an overly optimistic
    stochastic model. Conversely, values significantly smaller than one imply either an
    overly pessimistic stochastic model, i.e., underestimating the quality of
    observations, or an over-parameterized functional model.

    Parameters
    ----------
    rss : float
        The weighted residual sum of squares (rss).
    dof : int
        Degree of freedom.

    Returns
    -------
    float
        F-value.
    """
    return rss / dof


def read_knmi(file):
    """Precipitation and evapotranspiration data from
    The Royal Netherlands Meteorological Institute (KNMI).

    This function reads a KNMI file, extract daily precipitation (mm)
    and potential evapotranspiration (mm), and return it as a pd.DataFrame.

    The KNMI file can be downloaded through
    https://www.knmi.nl/nederland-nu/klimatologie/daggegevens.

    Parameters
    ----------
    file : str
        Path to the KNMI file with .txt format/

    Returns
    -------
    pd.DataFrame
        Daily precipitation and evapotranspiration.
    """
    df_meteo = pd.DataFrame(
        data=np.genfromtxt(
            file,
            delimiter=",",
            skip_header=53,
            missing_values="",
            filling_values=np.nan,
            usecols=(0, 1, 22, 40),
        ),
        columns=["meteo_id", "datum", "precip", "evapo"],
    )

    df_meteo["meteo_id"] = df_meteo["meteo_id"].astype(int)
    df_meteo["datum"] = pd.to_datetime(df_meteo["datum"].astype(str), format="%Y%m%d.0")
    df_meteo["precip"] = df_meteo["precip"] / 10  # mm
    df_meteo["evapo"] = df_meteo["evapo"] / 10  # mm

    return df_meteo


def format_with_uncertainty(value, uncertainty):
    """Format a value with its uncertainty.

    This function rounds the uncertainty to 1 significant figure
    and the value to match the decimal place of the uncertainty.

    Parameters:
    ----------
    value : float
        The main value.
    uncertainty : float
        The uncertainty associated with the value.

    Returns
    -------
    str
        A formatted string showing the value with its uncertainty.
    """
    if uncertainty == 0:
        return f"{value}"  # No uncertainty to display

    ## Determine the order of magnitude of the uncertainty
    order_of_magnitude = -int(math.floor(math.log10(abs(uncertainty))))

    ## Round the uncertainty to 1 significant figure
    rounded_uncertainty = round(uncertainty, order_of_magnitude)

    ## Round the value to match the decimal place of the rounded uncertainty
    rounded_value = round(value, order_of_magnitude)

    ## Format as a string
    formatted_value = f"{rounded_value:.{order_of_magnitude}f}"
    formatted_uncertainty = f"{rounded_uncertainty:.{order_of_magnitude}f}"

    return f"{formatted_value} $\pm$ {formatted_uncertainty}"
