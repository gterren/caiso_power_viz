# Joint Probabilistic Day-ahead Energy Forecast visualization companion

The Joint Probabilistic Day-ahead Energy Forecast model is available in ``https://github.com/gterren/caiso_power``. This repository contains the code developed for data download, processing, and result visualization.

## Data Download

* ``CAISO_API_for_renewable_generation_and_demand.py`` script to download from OASIS timeseries from solar generation, wind generation, and electricity demand in CAISO.
* ``NOAA_API_for_weather.py`` script to download NOAA HRRR Numerical Weather Forecasts (NWFs).
* ``consolidate_NOAA_and_CAISO.py`` script to consolidate NOAA HRRR NWFs with CAISO solar generation, wind generation, and electricity demand.

## Data Processing

* ``dimensionality_reduction.ipynb``

## Vizualizations

* ``motivation.ipynb`` motivation for improving an energy forecast (Fig. 1, and SI).
* ``system_level.ipynb`` baseline aggregated forecasts comparison (Fig2. 2, 5, 10, and SI).
* ``model_selection.ipynb`` model selection based on proposer scoring rules (Fig. 3 and SI).
* ``simulations.ipynb`` electricity demand, solar generation, and wind generation day-ahead forecast demonstration (Fig. 3, 5, 11, and SI),
* ``operational_reserves.ipynb`` dynamic operation reserves allocation experiments (Fig. 6).
* ``feature_maps.ipynb`` weather features selected by the different sparse learning methods (Fig. 9).
* ``input_data_viz.ipynb`` weather forecasts and reanalyzed measurements visualization (Fig. 7 and SI)

## Reference

The manuscript is currently undergoing revisions in Nature Communications. The draft is publicly available (https://www.researchsquare.com/article/rs-5891000/v1). We recommend using the following reference:

Terrén-Serrano, Guillermo, Ranjit Deshmukh, and Manel Martínez-Ramón. "Joint Probabilistic Day-Ahead Energy Forecast for Power System Operations." (2025).
