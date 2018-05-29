File description and organization:
A: input Data:
landsat_8.tif band 2-7 of landsat8 OLI imagery
dem_30.tif dem data from SRTM
svdnb_30.tif resampled SVIIRS/DNB imagery

B: Labelling/reference Data
beijing_global30r.tif Globeland30 MAP product of labelling data
lucc_2015.tif LUCC MAP of reference data
unchanged_10_15.tif agreement area of labelling data and reference data
XZQ_beijing.tif administrative grids of counties in Beijing
statistic.csv statistical data of Beijing counties aspect to the code in XZQ_beijing.tif

C: Processing scripts
train_no_water.py extracting sample data and train the CNN model
predict_lucc.py using the trained CNN to predict the LULC of whole region (beijing), the predict map is located in the directory of validate_5_11_v1/predict_lucc.txt
predict_weights.py fitting the weight algorithm with the average values at county level and create the weighting maps for population and GDP density. Weighting maps are saved in validate_5_11_v1/predict_weight14.txt
predict_county.py Disaggregating the population/GDP density of Beijing city and calculate the sum values with county units. Results are appended to the stastic.csv and saved to validate_5_11_v1/county_sta14.py
disaggregating.py cross-validate the accuracy of disaggregation results using the values of validate_5_11_v1/county_sta14.py and create the final disaggregating map  

