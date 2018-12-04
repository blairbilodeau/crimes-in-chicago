# Blair Bilodeau 1001232230
# STA2101 Final Project

## Libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import norm
from scipy.special import logsumexp
from colour import Color
from random import choice

plt.rcParams['agg.path.chunksize'] = 10000 # helps make plots with thousands of points

## Datasets

path = '/Users/blairbilodeau/Documents/Toronto/Current_Courses/STA2101/Project/Data/' # set to whatever path actually contains the data
crimes = pd.read_csv(path + 'Chicago_Crimes_2012_to_2017.csv') # https://www.kaggle.com/currie32/crimes-in-chicago#Chicago_Crimes_2012_to_2017.csv
schools = pd.read_csv(path + 'chicago-public-schools-high-school-progress-report-2013-2014.csv') # https://www.kaggle.com/chicago/chicago-public-schools-data#chicago-public-schools-high-school-progress-report-2013-2014.csv
police = pd.read_csv(path + 'police-stations.csv') # https://www.kaggle.com/chicago/chicago-police-stations#police-stations.csv

## Crimes Data Exploration

# We have 1 456 714 rows total
crimes = crimes[pd.notnull(crimes['X Coordinate'])] # 37 083 have NaN for location
crimes = crimes[crimes['X Coordinate'] > 0] # 77 have 0 for location
crimes = crimes[pd.notnull(crimes.Ward)] # 14 have NaN for ward
crimes = crimes[pd.notnull(crimes.District)] # 1 have NaN for district
crimes = crimes[pd.notnull(crimes['Community Area'])] # 22 have NaN for district
crimes = crimes[crimes.Year < 2017] # Only 30 crimes in 2017

# Clean up primary types
crimes = crimes.replace({'Primary Type':['NON - CRIMINAL', 'NON-CRIMINAL (SUBJECT SPECIFIED)']}, {'Primary Type':'NON-CRIMINAL'})
crimes = crimes.replace({'Primary Type':'OTHER NARCOTIC VIOLATION'}, {'Primary Type':'NARCOTICS'})
crimes = crimes.replace({'Primary Type':'CONCEALED CARRY LICENSE VIOLATION'}, {'Primary Type':'WEAPONS VIOLATION'})

# Time of event fields
crimes['Month'] = [int(date[0:2]) for date in crimes.Date]
crimes['Day'] = [int(date[3:5]) for date in crimes.Date]
crimes['Year'] = [int(date[6:10]) for date in crimes.Date] 

# Convert boolean to integer
crimes.Arrest = crimes.Arrest.astype(int)
crimes.Domestic = crimes.Domestic.astype(int)

# Lots of columns we don't care about
crimes.drop(['Unnamed: 0', 'Date', 'ID', 'Case Number', 'IUCR', 'Description', 'Location', 'Location Description', 'Updated On', 'Block', 'FBI Code'], axis=1, inplace=True)

## Crime Severity
# 1 - HOMICIDE, CRIM SEXUAL ASSAULT, KIDNAPPING, HUMAN TRAFFICKING, OFFENSE INVOLVING CHILDREN
# 2 - BURGLARY, THEFT, MOTOR VEHICLE THEFT, ROBBERY, ASSAULT, ARSON, SEX OFFENSE, BATTERY
# 3 - NARCOTICS, STALKING, WEAPONS VIOLATION, CRIMINAL DAMAGE, CRIMINAL TRESPASS
# 4 - GAMBLING, PROSTITUTION, OBSCENITY, LIQUOR LAW VIOLATION, PUBLIC PEACE VIOLATION, PUBLIC INDECENCY
# 5 - INTIMIDATION, INTERFERENCE WITH PUBLIC OFFICER, DECEPTIVE PRACTICE
# 6 - NON-CRIMINAL, OTHER OFFENSE

crime_severity = {'HOMICIDE':1, 'CRIM SEXUAL ASSAULT':1, 'KIDNAPPING':1, 'HUMAN TRAFFICKING':1, 
				  'OFFENSE INVOLVING CHILDREN':1, 'BURGLARY':2, 'THEFT':2, 'MOTOR VEHICLE THEFT':2, 'ROBBERY':2, 
				  'ASSAULT':2, 'ARSON':2, 'SEX OFFENSE':2, 'BATTERY':2, 'NARCOTICS':3, 'STALKING':3, 'WEAPONS VIOLATION':3, 
				  'CRIMINAL DAMAGE':3, 'CRIMINAL TRESPASS':3, 'GAMBLING':4, 'PROSTITUTION':4, 'OBSCENITY':4, 
				  'LIQUOR LAW VIOLATION':4, 'PUBLIC PEACE VIOLATION':4, 'PUBLIC INDECENCY':4, 'INTIMIDATION':5, 
				  'INTERFERENCE WITH PUBLIC OFFICER':5, 'DECEPTIVE PRACTICE':5, 'NON-CRIMINAL':6, 'OTHER OFFENSE':6}
crimes['Crime Severity'] = [crime_severity[crime] for crime in crimes['Primary Type']]
print(crimes.groupby('Crime Severity').agg('count').Latitude)

crime_severity_year = crimes.groupby(['Year', 'Crime Severity']).agg('count').Latitude
crime_year = crimes.groupby('Year').agg('count').Latitude
print(crime_severity_year.div(crime_year, level='Year')*100) # percentage of crimes by severity for each year

severity_y = crimes['Crime Severity']
severity_X = crimes[['Arrest', 'Domestic', 'Year', 'Latitude', 'Longitude', 'Month', 'Day']]
severity_model = sm.MNLogit(severity_y, severity_X).fit()
print(severity_model.summary()) # output of severity model results

## Compare crime to school quality

# We have 188 schools total
schools_cols = ['School ID', 'Student Response Rate', 'Teacher Response Rate', 'Involved Family', 
			   'Supportive Environment', 'Ambitious Instruction', 'Effective Leaders', 'Collaborative Teachers', 
			   'Safe', 'School Community', 'Parent-Teacher Partnership', 'Quality of Facilities', 
			   'Healthy Schools Certification', 'Creative Schools Certification', 'EPAS Growth Percentile', 
			   'EPAS Attainment Percentile', 'Grade ACT Attainment Percentile Grade 11', 'ACT Spring 2013 Average Grade 11', 
			   'Student Attendance Percentage 2013', 'One-Year DropOut Rate Percentage 2013', 'Latitude', 'Longitude']
schools = schools[schools_cols] # Only want to keep the columns that have a reasonable amount of data
schools = schools[list(~(np.array(np.isnan(schools['Student Attendance Percentage 2013']) & np.isnan(schools['One-Year DropOut Rate Percentage 2013']) & np.isnan(schools['ACT Spring 2013 Average Grade 11']))))] # remove 19 that have nearly no data
schools = schools[~np.isnan(schools['ACT Spring 2013 Average Grade 11']) & ~np.isnan(schools['EPAS Growth Percentile']) & ~np.isnan(schools['Grade ACT Attainment Percentile Grade 11']) & np.array(schools['One-Year DropOut Rate Percentage 2013'] > 0)] # drop 33 more schools with no ACT data

# Have to swap Longitude and Latitude
schools.rename(columns = {'Latitude': 'temp'}, inplace=True)
schools.rename(columns = {'Longitude':'Latitude', 'temp':'Longitude'}, inplace=True)

# View breakdown of categorical data
print(schools.groupby('Collaborative Teachers').agg('count')['School ID'])
print(schools.groupby('Safe').agg('count')['School ID']) # only useful one that has a reasonable percentage of missing values
print(schools.groupby('School Community').agg('count')['School ID'])
print(schools.groupby('Parent-Teacher Partnership').agg('count')['School ID'])
print(schools.groupby('Quality of Facilities').agg('count')['School ID'])

crimes['Latitude Radians'] = np.radians(crimes.Latitude)
crimes['Longitude Radians'] = np.radians(crimes.Longitude)

schools['Latitude Radians'] = np.radians(schools.Latitude)
schools['Longitude Radians'] = np.radians(schools.Longitude)

Earth_Radius = 6373 # approximate in km

# algorithm from https://andrew.hedges.name/experiments/haversine/ under MIT license
# returns array of Euclidean distance between school and each crime in crimes
def GeoDistSchool(school_lat, school_lon):

	lat_diff = crimes['Latitude Radians'] - school_lat
	lon_diff = crimes['Longitude Radians'] - school_lon

	a = np.square(np.sin(lat_diff / 2)) + np.cos(school_lat) * np.cos(crimes['Latitude Radians']) * np.square(np.sin(lon_diff / 2))
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

	return Earth_Radius * c

for school in schools.index:
	[school_lat, school_lon] = schools.loc[school, ['Latitude Radians', 'Longitude Radians']]
	distances = GeoDistSchool(school_lat, school_lon)
	schools.loc[school, 'Crimes Committed 0.25km'] = np.sum(distances <= 0.25)
	schools.loc[school, 'Crimes Committed 0.5km'] = np.sum(distances <= 0.5)
	schools.loc[school, 'Crimes Committed 0.75km'] = np.sum(distances <= 0.75)
	schools.loc[school, 'Crimes Committed 1km'] = np.sum(distances <= 1)
	schools.loc[school, 'Crimes Committed 1.5km'] = np.sum(distances <= 1.5)
	schools.loc[school, 'Crimes Committed 3km'] = np.sum(distances <= 3)
	schools.loc[school, 'Crimes Committed 5km'] = np.sum(distances <= 5)

# Fit multinomial logit model to predict 'Safe' categorization
schools_exists = schools[~(schools.Safe=='NOT ENOUGH DATA')] # 118 schools to build initial model
schools_exists_y = schools_exists.Safe
schools_exists_X = schools_exists[['Student Response Rate', 'Teacher Response Rate', 'EPAS Growth Percentile',
 								   'EPAS Attainment Percentile', 'Grade ACT Attainment Percentile Grade 11', 
 								   'ACT Spring 2013 Average Grade 11', 'Student Attendance Percentage 2013', 
 								   'One-Year DropOut Rate Percentage 2013', 'Latitude', 'Longitude', 
 								   'Crimes Committed 0.5km', 'Crimes Committed 1km']]
schools_exists_model = sm.MNLogit(schools_exists_y, schools_exists_X).fit() 
print(schools_exists_model.summary())

# Get rid of extraneous covariates and fit reduced model
schools_exists_X_reduced = schools_exists[['Student Response Rate', 'EPAS Attainment Percentile', 'One-Year DropOut Rate Percentage 2013', 'Latitude', 'Longitude']]
schools_exists_model_reduced = sm.MNLogit(schools_exists_y, schools_exists_X_reduced).fit() 
print(schools_exists_model_reduced.summary())

# Likelihood ratio test is 2(LL_full - LL_restricted)
G2 = 2 * (schools_exists_model.llf - schools_exists_model_reduced.llf)
df = schools_exists_model_reduced.df_resid - schools_exists_model.df_resid
pval = 1 - chi2.cdf(G2,df)

# Impute values of schools.Safe using reduced model

# Rows used to build the imputation model
schools_impute = schools[schools.Safe=='NOT ENOUGH DATA']
# Columns used to impute the data
schools_impute_X = schools_impute[['Student Response Rate', 'EPAS Attainment Percentile', 'One-Year DropOut Rate Percentage 2013', 'Latitude', 'Longitude']]

# Columns used to fit model on imputed data
schools_imputed_X = schools[['Student Response Rate', 'Teacher Response Rate', 'EPAS Growth Percentile', 'EPAS Attainment Percentile', 'Grade ACT Attainment Percentile Grade 11', 'ACT Spring 2013 Average Grade 11', 'Student Attendance Percentage 2013', 'One-Year DropOut Rate Percentage 2013', 'Latitude', 'Longitude', 'Crimes Committed 0.5km', 'Crimes Committed 1km']]
schools_imputed_conf_ints = []

# order of params output by model
# 0 - NEUTRAL
# 1 - STRONG
# 2 - VERY STRONG
# 3 - VERY WEAK
# 4 - WEAK
params = {0:'NEUTRAL', 1:'STRONG', 2:'VERY STRONG', 3:'VERY WEAK', 4:'WEAK'}

# Extracts parameters for distribution of coefficient estimates 
normal_95 = norm.ppf(0.975) # more decimals than 1.96
schools_exists_model_reduced_coefs = np.transpose(np.array(schools_exists_model_reduced.params)) # mean values for betas
schools_exists_model_reduced_se = np.zeros((4,5)) # standard deviations for betas
for i in range(4):
	for j in range(5):
		schools_exists_model_reduced_se[i,j] = (schools_exists_model_reduced._results.conf_int()[i][j][1] - schools_exists_model_reduced_coefs[i][j]) / normal_95

np.random.seed(123456) # set seed for reproducibility
# Multiple imputation ran 1000 times
num_iters = 1000
for iter in range(num_iters):

	# Matrix of sampled coefficient estimates
	# Each row corresponds to an outcome (response)
	# Each column corresponds to a covariate
	schools_exists_model_reduced_coefs_sample = np.zeros((4,5))
	for i in range(4):
		for j in range(5):
			# Sample each beta from the Normal distribution
			schools_exists_model_reduced_coefs_sample[i,j] = np.random.normal(schools_exists_model_reduced_coefs[i,j],schools_exists_model_reduced_se[i,j],1)

	# Compute L vector for each row to impute
	schools_imputed_temp = schools.copy()
	for school in schools_impute.index:
		L = np.dot(schools_exists_model_reduced_coefs_sample, schools_impute_X.loc[school]) # L1, L2, etc from lecture notes
		L_one = np.append(np.array([0]), L) # L for the reference value is 0, since need exp(L)=1
		log_denom = logsumexp(L_one) # computationally better than computing exponentials first and dividing
		probs = np.array([np.exp(L_one[i] - log_denom) for i in range(5)])
		schools_imputed_temp.loc[school, 'Safe'] = params[np.where(np.random.multinomial(1,probs) == 1)[0][0]]

	schools_imputed_temp_y = schools_imputed_temp.Safe
	schools_imputed_temp_model = sm.MNLogit(schools_imputed_temp_y, schools_imputed_X).fit(disp=0) 
	schools_imputed_conf_ints.append(schools_imputed_temp_model._results.conf_int())

# Now access the confidence intervals for each param and find average lower and upper bounds
schools_imputed_bootstrap_conf_ints = np.zeros((4,12,2))
for response in range(4):
	for coef in range(12):
		conf_lower = np.array([val[response][coef][0] for val in schools_imputed_conf_ints])
		conf_upper = np.array([val[response][coef][1] for val in schools_imputed_conf_ints])
		schools_imputed_bootstrap_conf_ints[response][coef][0] = np.mean(conf_lower)
		schools_imputed_bootstrap_conf_ints[response][coef][1] = np.mean(conf_upper)

np.set_printoptions(suppress=True)
print(np.round(schools_imputed_bootstrap_conf_ints,4)) # I manually put it into the nice format for the report
np.set_printoptions(suppress=False)

## Crime location compared to police stations plots

# Heatmap of crimes
ward_crime_percentages = crimes.groupby('Ward').agg('count').Latitude / len(crimes) * 100
colours = [str(col) for col in list(Color("lightsalmon").range_to(Color("darkred"),len(ward_crime_percentages)))]
sorted_crime_percentages = np.sort(ward_crime_percentages)
ward_colours = dict()
for index in range(len(colours)):
	ward_colours[ward_crime_percentages.index[np.where(ward_crime_percentages == sorted_crime_percentages[index])[0][0]]] = colours[index]
crimes['Ward Colour'] = [ward_colours[ward] for ward in crimes.Ward]

plt.figure('ward')
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.title('Crimes Heat Map by Ward + Police Stations', fontsize=12)
legend_handles = []
for index in range(len(colours)):
	colour = colours[index]
	plt.scatter(crimes[crimes['Ward Colour']==colour]['Latitude'], crimes[crimes['Ward Colour']==colour]['Longitude'], c=colour, s=2)
legend_handles = [mpatches.Patch(color=colours[i], label=str(round(sorted_crime_percentages[i],3)) + '%') for i in [0,9,19,29,39,49]]
plt.legend(handles = legend_handles)
plt.scatter(police.LATITUDE, police.LONGITUDE, c='navy', s=24)
plt.show('ward')

## Crimes in radius around police stations

police['Latitude Radians'] = np.radians(police.LATITUDE)
police['Longitude Radians'] = np.radians(police.LONGITUDE)

# Bootstrap locations

crimes_locations = list(zip(crimes['Latitude Radians'], crimes['Longitude Radians'])) # collection of all crime locations
num_crimes = len(crimes_locations)

num_bootstraps = 100
paired_mean_differences = np.zeros(num_bootstraps)

sample_size = len(crimes_locations) # experimented with not sampling the same number of crimes
sample_proportion = sample_size / len(crimes_locations)

np.random.seed(654321)

# Each bootstrap samples with replacement from the crime location daat
for bootstrap in range(num_bootstraps):

	sample_locations = [choice(crimes_locations) for sample in range(sample_size)]
	[sample_lat, sample_lon] = [np.array(tup) for tup in zip(*sample_locations)]

	# algorithm from https://andrew.hedges.name/experiments/haversine/ under MIT license
	# returns array of Euclidean distance between police stations and each crime in sampled crimes
	# redefined for each bootstrap since the sampled crime location data is an implicit argument
	def GeoDistPolice(police_lat, police_lon):

		lat_diff = sample_lat - police_lat
		lon_diff = sample_lon - police_lon

		a = np.square(np.sin(lat_diff / 2)) + np.cos(police_lat) * np.cos(sample_lat) * np.square(np.sin(lon_diff / 2))
		c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

		return Earth_Radius * c

	inner_radius = np.zeros(len(police.index))
	outer_radius = np.zeros(len(police.index))

	# Count crimes in both radii for each station
	for station in police.index:
		[police_lat, police_lon] = police.loc[station, ['Latitude Radians', 'Longitude Radians']]
		distances = GeoDistPolice(police_lat, police_lon)
		inner_radius[station] = np.sum(distances <= 1) / sample_proportion # to get them on the same scale as if all crimes were used
		outer_radius[station] = np.sum((distances > 1) & (distances <= np.sqrt(2))) / sample_proportion

	paired_mean_differences[bootstrap] = np.mean(inner_radius - outer_radius)

# Bootstrap confidence interval
print(np.percentile(paired_mean_differences,0.025))
print(np.percentile(paired_mean_differences,0.975))












