# pyspams
A Python implementation to create a realization of relative surface elevation using a set of estimated SPAMS parameters (Simple Parameterization for the Motion of Soils) and meteorological input data.

Reference: Conroy, P., van Diepen, S.A.N., Hanssen, R.F., 2023. SPAMS: A new empirical model for soft soil surface displacement based on meteorological input data. Geoderma 440, 116699. doi:10.1016/j.geoderma.2023.116699.

This model uses daily precipitation and evapotranspiration data, preferably from the closest meteorological station. The estimated SPAMS model parameters can be evaluated for chosen period.

## Requirements
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)

## Data
To generate time series surface motion using SPAMS, you need two datasets:
- A set of estimated SPAMS parameters (see the reference for more details).
- Daily precipitation and evapotranspiration values.

An example set of SPAMS parameters (SPAMS10) is available for several regions in the Netherlands through [4TU.ResearchData](https://doi.org/10.4121/c8646561-5475-4e23-bd90-3b1a3375ac7c.v1). Users can download the data and the metadata from that repository to model relative surface elevation changes using this repository. The meteorological data from The Royal Netherlands Meteorological Institute [KNMI](https://www.knmi.nl/nederland-nu/klimatologie/daggegevens) stations can be downloaded following the station ID available in the metadata.

## Usage
### Get information about required input parameters
```
python spams_main.py -h
```
Four parameters need to be specified: the path to the SPAMS10 parameters and meteorological directory and the start and end date of the desired period. Optionally, users can specify the KNMI file name and parcel ID.

### SPAMS time series surface motions
```
python spams_main.py --spams10_filepath <SPAMS10_FILEPATH> --meteo_dir <METEO_DIR> --start_date <START_DATE> --end_date <END_DATE>
```
Replace variables with < > to your desired.
