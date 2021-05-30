import covsirphy as cs
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import numba
import os
import pycountry as pyc
import pycountry_convert as pc
import random
import statsmodels.formula.api as smf
import statsmodels.api as sm
import wbgapi as wb
import seabornfig2grid as sfg
import matplotlib.gridspec as gridspec
import geopandas as gpd
import mapclassify
from tqdm import tqdm
from datetime import timedelta
from datetime import datetime
from scipy import stats
from scipy.integrate import odeint
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.ticker import FormatStrFormatter
from covid19dh import covid19
from IPython.display import display
from IPython.display import Image
pd.options.mode.chained_assignment = None 


# Total population, N.
N = 1000

# Initial number of infected, recovered and fatal individuals.
I0, R0, F0 = 1, 0, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - F0

# Effective contact rate, BETA, and mean recovery rate, GAMMA, (in 1/days).
ALPHA_1, ALPHA_2, BETA, GAMMA = .0002, .005, 0.2, .09 

# A grid of time points (in days).
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, ALPHA_1, ALPHA_2, BETA, GAMMA):
    S, I, R, F = y
    dSdt = -(N ** -1 * BETA * S * I)
    dIdt = N ** -1 * (1 - ALPHA_1) * BETA * S * I - (GAMMA + ALPHA_2) * I
    dRdt = GAMMA * I
    dFdt = N ** -1 * ALPHA_1 * BETA * S * I + ALPHA_2 * I
    return dSdt, dIdt, dRdt, dFdt

# Initial conditions vector
y0 = S0, I0, R0, F0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, ALPHA_1, ALPHA_2, BETA, GAMMA))
S, I, R, F = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax = plt.subplots(figsize=(20,10))
sns.lineplot(x = t, y = S/1000, label='Suseptible')
sns.lineplot(x = t, y = I/1000, label='Infected')
sns.lineplot(x = t, y = R/1000, label='Recovered with immunity')
sns.lineplot(x = t, y = F/1000, label='Fatal')

# Add figure annotations
ax.set_title('Time evolution of S(t), I(t), R(t) and F(t)', fontdict={'fontsize': 30})
ax.set_xlabel('Days from outbreak', fontsize=16)
ax.set_ylabel('Number of individuals (in THS)', fontsize=16)
ax.set_ylim(0, 1.1)
ax.tick_params(axis='both', labelsize=16)
ax.legend(fontsize=16)

ax.lines[0].set_linestyle('solid')
ax.lines[1].set_linestyle('dashed')
ax.lines[2].set_linestyle('dotted')
ax.lines[3].set_linestyle('dashdot')

plt.tight_layout()


scenarios = {'Mortality rate of uncategorized':  (0.0600, 0.005, 0.20, 0.075),
             'Mortality rate of infected':       (0.0002, 0.050, 0.20, 0.075),
             'Effective contact rate':           (0.0002, 0.005, 0.40, 0.075),
             'Recovery rate':                    (0.0002, 0.005, 0.20, 0.140)}

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(20,10))

for iIteration, iScenario in enumerate(scenarios):
    ALPHA_1, ALPHA_2, BETA, GAMMA = scenarios[iScenario]
    ret = odeint(deriv, y0, t, args=(N, ALPHA_1, ALPHA_2, BETA, GAMMA))
    df = pd.DataFrame(ret, columns=['Susceptible', 'Infected', 'Recovered with immunity', 'Fatal'])    
    sns.lineplot(data=df, ax=eval(f'ax{iIteration}'))
    eval(f"ax{iIteration}.set_title('{iScenario}')")

fig.tight_layout(pad=0.5)


BASE_PATH = r'C:\Users\Stephan\Documents\covid19_project'
os.chdir(BASE_PATH)
RAW_DATA_PATH = BASE_PATH + '\\data\\raw_data'
CLEAN_DATA_PATH = BASE_PATH + '\\data\\clean_data'
OUTPUT_PATH = BASE_PATH + '\\output'


rawCovidData = pd.read_csv(RAW_DATA_PATH + '\\covid19_dataset.csv', na_values=['NaN'], parse_dates=['date'], 
                           usecols=['id', 'date', 'confirmed', 'deaths', 'population', 'administrative_area_level_1'])

display(rawCovidData.loc[rawCovidData['id'] == 'FRA'].head())
display(rawCovidData.loc[rawCovidData['id'] == 'USA'].head())


# rawCovidData = covid19(raw=True, verbose=False)[0]
# rawCovidData.to_csv(RAW_DATA_PATH + '\\covid19_dataset.csv', na_rep='NaN', index=False)


rawCovidData.dtypes


rawCovidData = rawCovidData.rename(columns={'id':'country', 'administrative_area_level_1':'countryLong'})


if rawCovidData[['country', 'date', 'countryLong']].isnull().any(axis=None):
    raise ValueError('The variable country, date or countryLong contains missing values.')


startDate = rawCovidData['date'].loc[rawCovidData['date'] == rawCovidData['date'].min()].iloc[0] 
endDate = rawCovidData['date'].loc[rawCovidData['date'] == rawCovidData['date'].max()].iloc[0] - pd.Timedelta(11, unit='d')
rawCovidData = rawCovidData.loc[(rawCovidData['date'] >= startDate) & (rawCovidData['date'] <= endDate)]


rawCovidData[['country','date']].groupby(by=['country']).count().value_counts(normalize=True)


# Creation of a balanced panel dataset.
            
# dateList consists of all dates in the sample period and countryList
# contains all unique countries that exist in the dataset.
dateList = pd.date_range(startDate, endDate, freq='d')
countryList = rawCovidData['country'].unique()

# The following loop appends the missing observations to the dataset.
for iIteration, iCountry in enumerate(countryList):
    
    # Get data of country iCountry.
    iCountryData = rawCovidData.loc[rawCovidData['country'] == iCountry]
    
    # If for country iCountry an observation is missing, I create the observation 
    # and then append it to the dataset.
    for iDate in dateList:
        
        if iDate not in iCountryData['date'].values:
            
            missingObs = pd.DataFrame([[iCountry, iDate, np.nan, np.nan, iCountryData.iloc[0,4],
                                        iCountryData.iloc[0,5]]], columns=list(rawCovidData.columns.values))
            
            rawCovidData = rawCovidData.append(missingObs)


rawCovidData = rawCovidData.sort_values(by=['country', 'date'], ascending=[True, True])
rawCovidData = rawCovidData.set_index(['country', 'date'])


if len(rawCovidData['countryLong'].groupby(level=0).count().unique()) get_ipython().getoutput("= 1:")
    raise ValueError('The dataset is an unbalanced panel.')


if not(rawCovidData.index.is_unique):
    raise IndexError('Indices do not uniquely identify each observation.')


# Replacement and analysis of missing values.

# To replace and analyse the missing values efficiently, we use a numeric index variable in rawCovidData.
rawCovidData['indices'] = range(0, rawCovidData.shape[0]) 
variableList = ['confirmed', 'deaths']
printTitle = True

for iCountry in countryList:

    # Get data of country iCountry.
    iCountryData = rawCovidData.loc[rawCovidData.index.get_level_values('country') == iCountry]
    
    # This loop replaces all missing values that occur before the first non-Nan value with zero.
    for iVariable in variableList:
        
        # If the first value of the variable iVariable is not a NaN, 
        # then we do not replace any missing values with zero.
        if ~np.isnan(iCountryData[iVariable].iloc[0]):
            continue
        
        # If all values of variable iVariable are NaNs, then we replace all of them with zero.
        if iCountryData[iVariable].isnull().all():
            # Obtain index of last observation of country iCountry in rawCovidData. 
            lastIndex = iCountryData['indices'].iloc[-1] + 1
        # If the first value is a NaN and also non-NaN values exist in variable iVariable, 
        # then we get the index of the first non-NaN value.
        else:
            lastIndex = iCountryData['indices'].loc[iCountryData.index == iCountryData[iVariable].first_valid_index()].iloc[0]
            
        # Here we obtain the index in rawCovidData of the first observation of country iCountry.
        firstIndex = iCountryData['indices'].iloc[0]
        
        # Last, we replace all NaNs with zero.
        rawCovidData[iVariable].iloc[firstIndex:lastIndex] = 0
    
    # Analysis of missing values.    
    # Now we compute the fraction of NaNs between the first and last index of non-NaN values of the confirmed cases and 
    # deaths variables. Note that the fraction is also computed for countries that solely have NaNs.
    
    fracMissConfirmed = (1 - (iCountryData['confirmed'].loc[iCountryData['confirmed'].first_valid_index():iCountryData['confirmed'].last_valid_index()].count() / 
                            iCountryData['confirmed'].loc[iCountryData['confirmed'].first_valid_index():iCountryData['confirmed'].last_valid_index()].size))
    
    fracMissDeaths = (1 - (iCountryData['deaths'].loc[iCountryData['deaths'].first_valid_index():iCountryData['deaths'].last_valid_index()].count() / 
                         iCountryData['deaths'].loc[iCountryData['deaths'].first_valid_index():iCountryData['deaths'].last_valid_index()].size)) 
    
    # Here we list the countries for which more than 1% of confirmed cases or deaths are missing.
    if fracMissConfirmed > 0.01 or fracMissDeaths > 0.01:
        
        if printTitle:
            print('Countries with more than 1% missing values:')
            printTitle = False
            
        print(f"{iCountryData['countryLong'].iloc[0]}\n Percentage missing of confirmed cases: {round(fracMissConfirmed*100,2)}get_ipython().run_line_magic(""", "")
              f"\n Percentage missing of deaths: {round(fracMissDeaths*100,2)}get_ipython().run_line_magic("")", "")


rawCovidData['confirmed'] = rawCovidData['confirmed'].fillna(method='ffill')
rawCovidData['deaths'] = rawCovidData['deaths'].fillna(method='ffill')


for iVariable in variableList:

    isNotMonoIncr = rawCovidData[iVariable].groupby(level=0).diff() < 0
    idxIsNotMonoIncr = rawCovidData['indices'].loc[isNotMonoIncr]
    
    # We solve the monotonicity issue by using the following algorithm.
    # E.g., if the cumulative confirmed cases variable has the following chronologically ordered values 
    # 8 12 12 13 9 10 11 12, then we replace 12 12 13 with 9. Note that the 12 after the 9 is not replaced.
    for iIndex in idxIsNotMonoIncr:
        
        replacementValue = rawCovidData[iVariable].iloc[iIndex]
        iCountry = rawCovidData.index.get_level_values('country')[iIndex]
        iCountryData = rawCovidData.loc[rawCovidData.index.get_level_values('country') == iCountry]
        iCountryDataCut = iCountryData.loc[iCountryData['indices'] < iIndex]
        indicesToReplace = iCountryDataCut['indices'].loc[iCountryDataCut[iVariable] > replacementValue]
        
        for jIndex in indicesToReplace:
            rawCovidData[iVariable].loc[rawCovidData['indices'] == jIndex] = replacementValue


# COVID-19 variables.
if (~rawCovidData['confirmed'].between(0, 3.3e7)).any() or (~rawCovidData['deaths'].between(0, 5.9e5)).any():
    raise ValueError('The variables confirmed cases or deaths are not within a reasonable range.')

# Population variable.
if rawCovidData['population'].isnull().any() or (~rawCovidData['population'].between(400, 1.4e9)).any():
    raise ValueError('The variable population contains missing values or is not within a reasonable range.')


# Cumulative cases per 100,000 inhabitants.
rawCovidData['cumCasesPerHunThou'] = rawCovidData['confirmed'] / rawCovidData['population'] * 100000
# Cumulative deaths per 100,000 inhabitants.
rawCovidData['cumDeathsPerHunThou'] = rawCovidData['deaths'] / rawCovidData['population'] * 100000
# Daily new cases.
rawCovidData['dailyNewCases'] = rawCovidData['confirmed'].groupby(level=0).diff()
# Daily new deaths.
rawCovidData['dailyNewDeaths'] = rawCovidData['deaths'].groupby(level=0).diff()


if (rawCovidData['dailyNewCases'].dropna() < 0).any() or (rawCovidData['dailyNewDeaths'].dropna() < 0).any():
    raise ValueError('The variable dailyNewCases or dailyNewDeaths contains values that are smaller than zero.')


cleanCovidData = rawCovidData.drop(columns=['indices'])
cleanCovidData.to_csv(CLEAN_DATA_PATH + '\\covid19_clean_dataset.csv', na_rep='NaN', index=True)
del rawCovidData
del cleanCovidData 


cleanCovidData = pd.read_csv(CLEAN_DATA_PATH + '\\covid19_clean_dataset.csv', na_values=['NaN'], parse_dates=['date'], 
                             index_col=['country','date'])


get_ipython().run_cell_magic("capture", "", """fig, axes = plt.subplots(figsize=(22, 21), nrows=4, ncols=2)
fig.suptitle('Statistics of the COVID-19 pandemic', fontweight='bold', fontsize=36, y=0.95)
# The following two commands adjust the space between the subplots.
plt.subplots_adjust(wspace = 0.15) 
plt.subplots_adjust(hspace = 0.3)""")


globalCumCasesDeaths = cleanCovidData[['confirmed','deaths']].groupby(level=1).sum().reset_index()
globalCumCasesDeaths['weekday'] = globalCumCasesDeaths['date'].dt.dayofweek
globalCumCasesDeaths = globalCumCasesDeaths.loc[globalCumCasesDeaths['weekday'] == 6].set_index('date')


# Weekly global cumulative cases bar plot.
axes[0,0].bar(globalCumCasesDeaths.index, globalCumCasesDeaths['confirmed'], width=4, color='black')
axes[0,0].set_title('Weekly global cumulative cases', fontsize=20, fontweight='bold')
axes[0,0].tick_params(axis='both', labelsize=13)
axes[0,0].set_xlim([pd.Timestamp(2019, 12, 20), pd.Timestamp(2021, 5, 26)]) # Set the limits of the x axis.
axes[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0)) # Set y label to scientific notation.
exponent = axes[0,0].yaxis.get_offset_text() # Change the font size of the exponent (scientific notation) on the y axis.
exponent.set_size(13)

# Weekly global cumulative deaths bar plot.
axes[0,1].bar(globalCumCasesDeaths.index, globalCumCasesDeaths['deaths'], width=4, color='black')
axes[0,1].set_title('Weekly global cumulative deaths', fontsize=20, fontweight='bold')
axes[0,1].tick_params(axis='both', labelsize=13)
axes[0,1].set_xlim([pd.Timestamp(2019, 12, 20), pd.Timestamp(2021, 5, 26)])
axes[0,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
exponent = axes[0,1].yaxis.get_offset_text()
exponent.set_size(13)


worldMapData = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['iso_a3', 'continent', 'name', 'geometry']]


worldMapData['iso_a3'].loc[worldMapData['name'] == 'France'] = 'FRA'
worldMapData['iso_a3'].loc[worldMapData['name'] == 'Norway'] = 'NOR'
worldMapData['iso_a3'].loc[worldMapData['name'] == 'Kosovo'] = 'RKS'
worldMapData['iso_a3'].loc[worldMapData['name'] == 'Somaliland'] = 'Somaliland'
worldMapData['iso_a3'].loc[worldMapData['name'] == 'N. Cyprus'] = 'N. Cyprus'


worldMapData = worldMapData.loc[worldMapData["name"] get_ipython().getoutput("= 'Antarctica']")


countryList = cleanCovidData.index.get_level_values('country').unique()
for iCountry in countryList:
    
    if iCountry == countryList[0]:
        
        print('The following countries are not included in the geopandas world map:')
    
    if iCountry not in worldMapData['iso_a3'].values:
        
        countryNameLong = cleanCovidData['countryLong'].loc[cleanCovidData.index.get_level_values('country') == iCountry].iloc[0]
        print(countryNameLong)


cleanCovidDataMerge = cleanCovidData.reset_index()[['country', 'date', 'dailyNewCases', 'dailyNewDeaths', 'cumCasesPerHunThou', 'cumDeathsPerHunThou']]
worldMapData = worldMapData.merge(cleanCovidDataMerge, left_on='iso_a3', right_on='country', how='left', validate='1:m')


missCountriesInC19DH = worldMapData['name'].loc[worldMapData['country'].isnull()]
print(f'The following countries are not included in the C19DH:\n{missCountriesInC19DH.to_string(index=False)}')


# Weekly new cases per continent.
casesPerContinent = worldMapData[['continent', 'date', 'dailyNewCases']].groupby(['continent', 'date']).sum()
casesPerContinent = casesPerContinent.pivot_table(index=['date'], columns=['continent'], values='dailyNewCases')
casesPerContinent = casesPerContinent.resample('W').sum()

# Weekly new deaths per continent.
deathsPerContinent = worldMapData[['continent', 'date', 'dailyNewDeaths']].groupby(['continent', 'date']).sum()
deathsPerContinent = deathsPerContinent.pivot_table(index=['date'], columns=['continent'], values='dailyNewDeaths')
deathsPerContinent = deathsPerContinent.resample('W').sum()


# Weekly new cases per continent stacked bar plot.
continentList = list(casesPerContinent.columns.values)
bottom = np.zeros(casesPerContinent['Africa'].size)
# Here we stack the bars of each continent.
for iContinent in continentList:
    axes[1,0].bar(casesPerContinent.index, casesPerContinent[iContinent], width=4, bottom=bottom)
    bottom += casesPerContinent[iContinent]
    
axes[1,0].set_title('Weekly new cases', fontsize=20, fontweight='bold')
axes[1,0].legend(['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'], loc='upper left', fontsize=13)
axes[1,0].tick_params(axis='both', labelsize=13)
axes[1,0].set_xlim([pd.Timestamp(2019, 12, 20), pd.Timestamp(2021, 5, 26)])
axes[1,0].set_ylim([0, 6*1e6]) # Zero was not displayed on y axis, hence I add it with set_ylim.
axes[1,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
exponent = axes[1,0].yaxis.get_offset_text()
exponent.set_size(13)

# Weekly new deaths per continent stacked bar plot.
continentList = list(deathsPerContinent.columns.values)
bottom = np.zeros(deathsPerContinent['Africa'].size)
for iContinent in continentList:
    axes[1,1].bar(deathsPerContinent.index, deathsPerContinent[iContinent], width=4, bottom=bottom)
    bottom += deathsPerContinent[iContinent]
    
axes[1,1].set_title('Weekly new deaths', fontsize=20, fontweight='bold')
axes[1,1].legend(['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'], loc='upper left', fontsize=13)
axes[1,1].tick_params(axis='both', labelsize=13)
axes[1,1].set_xlim([pd.Timestamp(2019, 12, 20), pd.Timestamp(2021, 5, 26)])
axes[1,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
exponent = axes[1,1].yaxis.get_offset_text()
exponent.set_size(13)


# Here we extract all observations of the 16 May 2021. Note that I also keep the countries 
# that do not exist in the C19DH because they will be display as missing on the world map (grey colored).
currentWorldMap = worldMapData.loc[(worldMapData['date'] == pd.Timestamp(2021, 5, 16)) | (worldMapData['date'].isnull())]

# Map plot 1: Total cases per hundred thousand inhabitants of a country.
currentWorldMap.plot(ax=axes[2,0], column='cumCasesPerHunThou', cmap='autumn_r', scheme='Quantiles', k=6, legend=True, 
                  legend_kwds=dict(loc='lower left'), missing_kwds={"color": "grey", "label": "No data available"})
axes[2,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)  
axes[2,0].set_title('Cases per 100,000 inhabitants', fontsize=20, fontweight='bold')

# Map plot 2: Total deaths per hundred thousand inhabitants of a country.
currentWorldMap.plot(ax=axes[2,1], column='cumDeathsPerHunThou', cmap='copper_r', scheme='Quantiles', k=6, legend=True, 
                  legend_kwds=dict(loc='lower left'), missing_kwds={"color": "grey", "label": "No data available"})
axes[2,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)  
axes[2,1].set_title('Deaths per 100,000 inhabitants', fontsize=20, fontweight='bold')


countryList = ['DEU', 'FRA', 'RUS', 'USA']
linestyleList = ['-', '--', '-.', ':']

# Daily new cases per country plot.
for iValue, iCountry  in enumerate(countryList):
    iCountryData = cleanCovidData.loc[cleanCovidData.index.get_level_values('country') == iCountry]
    iCountryData['casesMovAvg'] = (iCountryData['dailyNewCases'] / iCountryData['population'] * 100000).rolling(window=14).mean()
    axes[3,0].plot(iCountryData.index.get_level_values('date'), iCountryData['casesMovAvg'], linestyle=linestyleList[iValue], linewidth=3)

axes[3,0].set_title('Daily cases per 100,000 inhabitants', fontsize=20, fontweight='bold')
axes[3,0].legend(['Germany', 'France', 'Russia', 'United States'], loc='upper left', fontsize=13)
axes[3,0].tick_params(axis='both',labelsize=13)
axes[3,0].set_xlim([pd.Timestamp(2019, 12, 20), pd.Timestamp(2021, 5, 26)])

# Daily new deaths per country plot.
for iValue, iCountry  in enumerate(countryList):
    iCountryData = cleanCovidData.loc[cleanCovidData.index.get_level_values('country') == iCountry]
    iCountryData['deathsMovAvg'] = (iCountryData['dailyNewDeaths'] / iCountryData['population'] * 100000).rolling(window=14).mean()
    axes[3,1].plot(iCountryData.index.get_level_values('date'), iCountryData['deathsMovAvg'], linestyle=linestyleList[iValue], linewidth=3)
    
axes[3,1].set_title('Daily deaths per 100,000 inhabitants', fontsize=20, fontweight='bold')
axes[3,1].legend(['Germany', 'France', 'Russia', 'United States'], loc='upper right', fontsize=13)
axes[3,1].tick_params(axis='both',labelsize=13)
axes[3,1].set_xlim([pd.Timestamp(2019, 12, 20), pd.Timestamp(2021, 5, 26)])


pathAndFilename = OUTPUT_PATH + '/desc_stats_covid19.png'
fig.savefig(pathAndFilename, bbox_inches='tight', dpi=150)

Image(filename=pathAndFilename) 


dataLoader = cs.DataLoader("input")
jhuData = dataLoader.jhu(verbose=0)
populationData = dataLoader.population()
oxcgrtData = dataLoader.oxcgrt()

df = jhuData.raw.drop_duplicates('Country')
countriesDict = dict(zip(df['ISO3'], df['Country']))


# Defining the directory to store the estimated parameters
backupfileDict = cs.Filer(directory='retrospective')

files = [x[:3] for x in os.listdir('retrospective')]

for iCountry in tqdm(countriesDict):
    
    if iCountry not in files:
        
        try:
            
            # Instantiation of the country object.
            exec(f'{iCountry} = cs.Scenario(country="{countriesDict[iCountry]}")')
            eval(iCountry).register(jhuData, extras=[oxcgrtData])

            # S-R relationship analysis
            eval(iCountry).trend(show_figure = False)

            # Minimum of 90 days for parameter estimation.
            df = eval(iCountry).records(variables='all', show_figure=False).tail(-90)

            # Identification of largest stringency increase.
            iMax = np.argmax(np.diff(df['Stringency_index']))
            date = df.iloc[iMax, :]['Date']
            row = eval(iCountry).summary()[eval(iCountry).summary().apply(lambda k: datetime.strptime(k[1],'get_ipython().run_line_magic("d%b%Y')", " <= date <= datetime.strptime(k[2],'%d%b%Y'), axis=1)]")
            beginningDate = row['End'][0]

            # Parameter estimation
            eval(iCountry).retrospective(beginning_date=beginningDate, model=cs.SIRF, control="Main", target="Retrospective", timeout=180)
            
            # JSON serialization
            eval(iCountry).backup(**backupfileDict.json(f'{iCountry}'))

        except:
            
            continue


retrospective = {}

for iFile in tqdm(os.listdir('../data/retrospective')):
    
    if iFile.endswith('json'):
        
        country = iFile[:3]
        
        try:
            
            exec(f'{country} = cs.Scenario(country="{countriesDict[country]}")')
            eval(country).register(jhuData, extras=[oxcgrtData])
            eval(country).restore(f'../data/retrospective/{country}.json')
            
            continent = pc.country_alpha2_to_continent_code(pyc.countries.get(alpha_3=country).alpha_2)
            
            dic = retrospective.get(continent, {})
            dic.update({country:eval(country)})
            retrospective[continent] = dic

        except:
            
            continue


cols = {'EU':'Europe', 'AS':'Asia', 'NA':'North America', 'SA':'South America'}
rows = ['Confirmed cases'] * 4

fig, axes = plt.subplots(4, 4, figsize=(30,20))

for iIteration, iCol in enumerate(cols):

    for jIteration, iCountry in enumerate(random.sample(retrospective[iCol].keys(), 4)):
        
        df = eval(iCountry).history('Confirmed', show_figure=False).reset_index()
        sns.lineplot(data=df, x='Date', y='Actual', ax=axes[jIteration][iIteration], label='Actual cases')
        sns.lineplot(data=df, x='Date', y='Retrospective', ax=axes[jIteration][iIteration], label='Model projection')
        axes[jIteration][iIteration].set_ylabel(None)
        axes[jIteration][iIteration].set_xlabel(countriesDict[iCountry], fontsize=20)
        axes[jIteration][iIteration].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        exponent = axes[jIteration][iIteration].yaxis.get_offset_text()
        exponent.set_size(13)

for iAx, iCol in zip(axes[0], cols):
    iAx.set_title(cols[iCol], {'weight': 'bold', 'fontsize':20})

for iAx, iRow in zip(axes[:,0], rows):
    iAx.set_ylabel(iRow, rotation=90, fontsize = 20)

fig.suptitle("Variable 'Confirmed': Model projection vs actual cases by continent", size=30, y=1.001)
fig.tight_layout(h_pad=0.3, w_pad=10)
plt.show()


cols = {'EU':'Europe', 'AS':'Asia', 'NA':'North America', 'SA':'South America'}
rows = ['Fatal cases'] * 4

fig, axes = plt.subplots(4, 4, figsize=(30,20))

for iIteration, iCol in enumerate(cols):

    for jIteration, iCountry in enumerate(random.sample(retrospective[iCol].keys(), 4)):
        
        df = eval(iCountry).history('Fatal', show_figure=False).reset_index()
        sns.lineplot(data=df, x='Date', y='Actual', ax=axes[jIteration][iIteration], label='Actual cases')
        sns.lineplot(data=df, x='Date', y='Retrospective', ax=axes[jIteration][iIteration], label='Model projection')
        axes[jIteration][iIteration].set_ylabel(None)
        axes[jIteration][iIteration].set_xlabel(countriesDict[iCountry], fontsize=20)
        axes[jIteration][iIteration].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        exponent = axes[jIteration][iIteration].yaxis.get_offset_text()
        exponent.set_size(13)

for iAx, iCol in zip(axes[0], cols):
    iAx.set_title(cols[iCol], {'weight': 'bold', 'fontsize':20})

for iAx, iRow in zip(axes[:,0], rows):
    iAx.set_ylabel(iRow, rotation=90, fontsize = 20)

fig.suptitle("Variable 'Fatal': Model projection vs actual cases by continent", size=30, y=1.001)
fig.tight_layout(h_pad=0.3, w_pad=10)
plt.show()


cols = {'EU':'Europe', 'AS':'Asia', 'NA':'North America', 'SA':'South America'}
rows = ['Recovered cases'] * 4

fig, axes = plt.subplots(4, 4, figsize=(30,20))

for iIteration, iCol in enumerate(cols):

    for jIteration, iCountry in enumerate(random.sample(retrospective[iCol].keys(), 4)):
        
        df = eval(iCountry).history('Recovered', show_figure=False).reset_index()
        sns.lineplot(data=df, x='Date', y='Actual', ax=axes[jIteration][iIteration], label='Actual cases')
        sns.lineplot(data=df, x='Date', y='Retrospective', ax=axes[jIteration][iIteration], label='Model projection')
        axes[jIteration][iIteration].set_ylabel(None)
        axes[jIteration][iIteration].set_xlabel(countriesDict[iCountry], fontsize=20)
        axes[jIteration][iIteration].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        exponent = axes[jIteration][iIteration].yaxis.get_offset_text()
        exponent.set_size(13)

for iAx, iCol in zip(axes[0], cols):
    iAx.set_title(cols[iCol], {'weight': 'bold', 'fontsize':20})

for iAx, iRow in zip(axes[:,0], rows):
    iAx.set_ylabel(iRow, rotation=90, fontsize = 20)

fig.suptitle("Variable 'Recovered': Model projection vs actual cases by continent", size=30, y=1.001)
fig.tight_layout(h_pad=0.3, w_pad=10)
plt.show()


window = 90


analysis = {}
    
for iCountry in tqdm(countriesDict):
    
    try:
        
        df = eval(iCountry).records(variables='all', show_figure=False).tail(-90)
        iMax = np.argmax(np.diff(df['Stringency_index']))
        date = df.iloc[iMax, :]['Date']
        row = eval(iCountry).summary()[eval(iCountry).summary().apply(lambda k: datetime.strptime(k[1],'get_ipython().run_line_magic("d%b%Y')", " <= date <= datetime.strptime(k[2],'%d%b%Y'), axis=1)]")
        beginningDate = row['End'][0]

        df = eval(iCountry).summary()
        endDate = df.loc['Main'].iloc[-1]['End']

        if (datetime.strptime(endDate, 'get_ipython().run_line_magic("d%b%Y')", " - datetime.strptime(beginningDate, '%d%b%Y')).days >= window:")
            analysis[iCountry] = datetime.strptime(beginningDate, 'get_ipython().run_line_magic("d%b%Y')", "")
    except:
        
        continue


dependentVariable = {}
    
for iCountry in tqdm(analysis):

    dependentVariable[iCountry] = {}

    try:
        
        dfConfirmed = eval(iCountry).history(target='Confirmed', show_figure=False)[['Actual', 'Retrospective']].loc[analysis[iCountry]:analysis[iCountry]+timedelta(days=window)]
        dfInfected = eval(iCountry).history(target='Infected', show_figure=False)[['Actual', 'Retrospective']].loc[analysis[iCountry]:analysis[iCountry]+timedelta(days=window)]
        dfFatal = eval(iCountry).history(target='Fatal', show_figure=False)[['Actual', 'Retrospective']].loc[analysis[iCountry]:analysis[iCountry]+timedelta(days=window)]
        dfRecovered = eval(iCountry).history(target='Recovered', show_figure=False)[['Actual', 'Retrospective']].loc[analysis[iCountry]:analysis[iCountry]+timedelta(days=window)]

        dependentVariable[iCountry]['Confirmed'] = dfConfirmed
        dependentVariable[iCountry]['Infected'] = dfInfected
        dependentVariable[iCountry]['Fatal'] = dfFatal
        dependentVariable[iCountry]['Recovered'] = dfRecovered

    except:
        continue


dfs = ['Confirmed', 'Infected', 'Fatal', 'Recovered']

for iDf in dfs:
    
    indices = []
    
    exec(f"{iDf}_diffs = []")
    exec(f"{iDf} = pd.DataFrame()")

    for iCountry in analysis:

        exec(f"{iDf}_diff = ((np.log(dependentVariable[iCountry]['{iDf}']['Actual'] + 1) - np.log(dependentVariable[iCountry]['{iDf}']['Retrospective'] + 1))**2).sum()")
        indices.append(iCountry)
        exec(f"{iDf}_diffs.append({iDf}_diff)")

    exec(f"{iDf}['Difference'] = {iDf}_diffs")
    exec(f"{iDf}['Country'] = indices")
    exec(f"{iDf}['Continent'] = {iDf}['Country'].apply(lambda k: pc.country_alpha2_to_continent_code(pyc.countries.get(alpha_3=k).alpha_2) if pyc.countries.get(alpha_3=k) get_ipython().getoutput("= None else k).values")")
    exec(f"{iDf} = {iDf}.set_index(['Continent', 'Country'])")
    exec(f"{iDf} = {iDf}.sort_index()")


fig, axes = plt.subplots(2,2, figsize=(20,15))

for iIteration, iAx in enumerate(axes.flat):

    sns.barplot(data=eval(dfs[iIteration]).reset_index().sort_values('Difference', ascending=False).head(30), x='Country', y='Difference', hue='Continent', dodge=False, ax=iAx, palette=sns.color_palette('mako_r', 5))
    iAx.set_xticklabels(eval(dfs[iIteration]).reset_index().sort_values('Difference', ascending=False).head(30)['Country'].apply(lambda k: pyc.countries.get(alpha_3=k).name))
    iAx.tick_params(axis='x', labelrotation=90)
    iAx.set_title(f"Variable '{dfs[iIteration]}': Difference between model projections and actual cases by country", size=15)
    iAx.set_ylabel('Difference', size=15) 
    iAx.set_xlabel(None) 
    
fig.tight_layout(pad=0.8)


fig, axes = plt.subplots(2,2, figsize=(20,15))

for iIteration, iAx in enumerate(axes.flat):

    sns.barplot(data=eval(dfs[iIteration]).reset_index().sort_values('Difference', ascending=False).groupby('Continent').sum().reset_index(), x='Continent', y='Difference', ax=iAx, errwidth=0, palette=sns.color_palette('mako_r', 5))
    iAx.set_xticklabels(eval(dfs[iIteration]).reset_index().sort_values('Difference', ascending=False).groupby('Continent').sum().reset_index()['Continent'].apply(lambda k: pc.convert_continent_code_to_continent_name(k)))
    iAx.tick_params(axis='x', labelrotation=90)
    iAx.set_title(f"Variable '{dfs[iIteration]}': Difference between model projections and actual cases by continent", size=15, y=1.005)
    iAx.set_ylabel('Difference', size=15)
    iAx.set_xlabel(None)
    
fig.tight_layout(pad=0.8)


for iIteration, iDf in enumerate(dfs):
    exec(f"g{iIteration} = sns.catplot(data=eval(iDf).reset_index().sort_values('Difference'), x='Continent', y='Difference', height=7, palette=sns.color_palette('mako_r', 5))")

fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(2, 2)

for iIteration, iGrid in enumerate(gs):
    exec(f"mg{iIteration} = sfg.SeabornFig2Grid(g{iIteration}, fig, iGrid)")
    
for iIteration, iAx in enumerate(fig.axes):
    iAx.set_title(f"{dfs[iIteration]} cases")
    iAx.set_xticklabels(eval(dfs[iIteration]).reset_index().sort_values('Difference', ascending=False).groupby('Continent').sum().reset_index()['Continent'].apply(lambda k: pc.convert_continent_code_to_continent_name(k)))
    iAx.set_xlabel(None)
    
gs.tight_layout(fig, pad=0.8)
fig.suptitle('Distribution of the dependent variable by continent (scatter)', size=15, y=1.05)
plt.show()


for iIteration, iDf in enumerate(dfs):
    exec(f"g{iIteration} = sns.catplot(data=Infected.reset_index().sort_values('Difference'), x='Continent', y='Difference', kind='box', height=7, palette=sns.color_palette('mako_r', 5))")

fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(2, 2)

for iIteration, iGrid in enumerate(gs):
    exec(f"mg{iIteration} = sfg.SeabornFig2Grid(g{iIteration}, fig, iGrid)")
    
for iIteration, iAx in enumerate(fig.axes):
    iAx.set_title(f"{dfs[iIteration]} cases")
    iAx.set_xticklabels(eval(dfs[iIteration]).reset_index().sort_values('Difference', ascending=False).groupby('Continent').sum().reset_index()['Continent'].apply(lambda k: pc.convert_continent_code_to_continent_name(k)))
    iAx.set_xlabel(None)
    
gs.tight_layout(fig, pad=0.8)
fig.suptitle('Distribution of the dependent variable by continent (boxplot)', size=15, y=1.05)
plt.show()


# Import of quality of governance dataset
qdf = pd.read_excel('../data/raw_data/qog_std_cs_jan21.xlsx')
qdf = qdf.set_index('ccodealp').drop(columns=['ccode', 'cname', 'ccodewb', 'version', 'ccodecow'])

cols = set([iCol.split('_')[0] for iCol in qdf.columns])
datasets = {}

for iCol in cols:
    datasets[iCol] = {}
    datasets[iCol]['regressors'] = qdf[qdf.columns[qdf.columns.str.startswith(iCol)]]


modelsDict = {}


reg = pd.merge(Confirmed, datasets['wdi']['regressors'], left_on=['Country'], right_index=True)
reg = reg.applymap(lambda k: pd.to_numeric(k, errors='coerce'))
reg = reg.replace([np.inf, -np.inf], np.nan)
reg = reg.fillna(reg.median())
Y = reg['Difference'].to_frame()
X = reg.iloc[:, 1:]

selector = SelectKBest(f_regression, k=3)
selector.fit_transform(X, Y)
Xnew = X.iloc[:,selector.get_support(indices=True)]

reg = reg[['Difference']+ Xnew.columns.tolist()]

model = sm.OLS(Y, sm.add_constant(Xnew)).fit()
modelsDict['Linear model'] = {}
modelsDict['Linear model']['model'] = model
modelsDict['Linear model']['reg'] = reg
model.summary()


continentsDict = ['AF', 'AS', 'EU', 'NA', 'SA']
dummiesList = [f'Continent_{c}' for c in continentsDict]


reg = pd.merge(Confirmed, datasets['wdi']['regressors'], left_on=['Country'], right_index=True)
reg = reg.applymap(lambda k: pd.to_numeric(k, errors='coerce'))
reg = reg.replace([np.inf, -np.inf], np.nan)
reg = reg.fillna(reg.median())
Y = reg['Difference'].to_frame()
X = reg.iloc[:, 1:]

selector = SelectKBest(f_regression, k=3)
selector.fit_transform(X, Y)
Xnew = X.iloc[:,selector.get_support(indices=True)]

reg = reg[['Difference']+ Xnew.columns.tolist()]

model = smf.ols(formula = 'Difference ~ ' + ' + '.join(Xnew.columns) + ' + ' + ' + '.join(dummiesList), data=pd.get_dummies(reg.reset_index(), columns=['Continent'])).fit()
modelsDict['Fixed effects model'] = {}
modelsDict['Fixed effects model']['model'] = model
modelsDict['Fixed effects model']['reg'] = reg
model.summary()


reg = pd.merge(Confirmed, datasets['gggi']['regressors'], left_on=['Country'], right_index=True)
reg = reg.applymap(lambda k: pd.to_numeric(k, errors='coerce'))
reg = reg.replace([np.inf, -np.inf], np.nan)
reg = reg.fillna(reg.median())
Y = reg['Difference'].to_frame()
X = reg.iloc[:, 1:]

Xint = pd.DataFrame(PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(X), columns = list(X.columns) + list(itertools.combinations(X.columns, 2)))
Xint.index = Y.index
selector = SelectKBest(f_regression, k=1)
selector.fit_transform(Xint, Y)
Xint = Xint.iloc[:,selector.get_support(indices=True)]
Xint.columns = [x[0] + '_' + x[1] if type(x) == tuple else x for x in Xint.columns]

reg = pd.merge(Y, Xint, left_index=True, right_index=True)

model = smf.ols(formula = 'Difference ~ ' + ' + '.join(Xint.columns), data=reg.reset_index()).fit()
modelsDict['Interaction model'] = {}
modelsDict['Interaction model']['model'] = model
modelsDict['Interaction model']['reg'] = reg
model.summary()


fig, axes = plt.subplots(1, 3, figsize=(20,5))

for iIteration, iModel in enumerate(modelsDict):
    
    sm.qqplot(modelsDict[iModel]['model'].resid, stats.t, distargs=(4,), fit=True, ax=axes[iIteration], line='45')
    axes[iIteration].set_title(iModel)

fig.suptitle('Normal Q-Q plot', size=20)
fig.tight_layout(pad=0.8)


fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for iIteration, iModel in enumerate(modelsDict):
    
    sns.regplot(modelsDict[iModel]['model'].fittedvalues, 
           np.sqrt(np.abs(modelsDict[iModel]['model'].get_influence().resid_studentized_internal)), 
            scatter=True, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8},
          scatter_kws={'facecolors':'none', 'edgecolors':'black'}, ax=axes[iIteration])
    axes[iIteration].set_title(iModel)
    axes[iIteration].set_xlabel('Fitted Values')
    axes[iIteration].set_ylabel('$\sqrt{|Standardized Residuals|}$')

fig.suptitle('Scale-Location test', size=20)
fig.tight_layout(pad=0.8)


g0 = sns.PairGrid(modelsDict['Linear model']['reg'].reset_index(), hue='Continent', palette=sns.color_palette('mako_r', 5)).map(sns.scatterplot)
g1 = sns.PairGrid(modelsDict['Fixed effects model']['reg'].reset_index(), hue='Continent', palette=sns.color_palette('mako_r', 5)).map(sns.scatterplot)
g2 = sns.PairGrid(modelsDict['Interaction model']['reg'].reset_index(), hue='Continent', palette=sns.color_palette('mako_r', 5)).map(sns.scatterplot)

fig = plt.figure(figsize=(20,8))
gs = gridspec.GridSpec(1, 3)

mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
mg2 = sfg.SeabornFig2Grid(g2, fig, gs[2])

gs.tight_layout(fig, pad=2)
fig.suptitle('Multicollinearity test', size=20, y=1.05)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for iIteration, iModel in enumerate(modelsDict):
    
    sns.residplot(modelsDict[iModel]['model'].fittedvalues, 'Difference', data=modelsDict[iModel]['reg'], lowess=True, 
                  scatter_kws={'facecolors':'none', 'edgecolors':'black'}, line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8}, ax=axes[iIteration])  

fig.suptitle('Heteroscedasticity tests', size=20)
fig.tight_layout(pad=0.8)


# Basis scenario
country = 'CHE'
eval(country).add(days=360, name="Basis")

# Lockdown scenario
rhoLockdown = eval(country).get("rho", phase="last") * 0.5
eval(country).add(days=360, name="Lockdown", rho=rhoLockdown)

# Medicine scenario
kappaMedicine = eval(country).get("kappa", phase="last") * 0.5
sigmaMedicine = eval(country).get("sigma", phase="last") * 2
eval(country).add(days=360, name="Medicine", kappa=kappaMedicine, sigma=sigmaMedicine)

# Vaccine scenario
rhoVaccine = eval(country).get("rho", phase="last") * 0.8
kappaVaccine = eval(country).get("kappa", phase="last") * 0.6
sigmaVaccine = eval(country).get("sigma", phase="last") * 1.2
eval(country).add(days=360, name="Vaccine",  rho=rhoVaccine, kappa=kappaVaccine, sigma=sigmaVaccine)


# Simulate the number of cases
variables = {'C':'Confirmed', 'I':'Infected', 'F':'Fatal', 'R':'Recovered'}
scenarios = ['Basis', 'Lockdown', 'Medicine', 'Vaccine']

for iScenario in scenarios:
    exec(f"{iScenario} = eval(country).simulate(name='{iScenario}', variables = ['Confirmed', 'Infected', 'Fatal', 'Recovered'], show_figure=False)")

for iVariable in variables:
    
    exec(f"{iVariable} = eval(scenarios[0])[[variables[iVariable], 'Date']]")
    exec(f"{iVariable}.columns = ['Basis', 'Date']")
    
    for iScenario in scenarios[1:]:
        to_merge = eval(iScenario)[[variables[iVariable], 'Date']]
        to_merge.columns = [iScenario, 'Date']
        exec(f"{iVariable} = pd.merge({iVariable}, to_merge, left_on='Date', right_on='Date')")
    


fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(20, 12), sharex=True)

for iIteration, iVariable in enumerate(variables):
    
    df = eval(iVariable).tail(360)
    sns.lineplot(x='Date', y='value', hue='variable', data=pd.melt(df, ['Date']), ax=eval(f'ax{iIteration}'), palette=sns.color_palette("mako_r", 4), sizes=10, linewidth=2)
    eval(f'ax{iIteration}.set_title("Scenarios of variable: {variables[iVariable]}", size=20)')
    eval(f'ax{iIteration}.set_ylabel(None)')
    eval(f'ax{iIteration}.ticklabel_format(axis="y", style="sci", scilimits=(0,0))')
    exec(f'exponent = ax{iIteration}.yaxis.get_offset_text()')
    eval(f'exponent.set_size(13)')
    
    for iLinestyle, iLine in eval(f"zip(('solid', 'dotted', 'dashed', 'dashdot'), ax{iIteration}.lines)"):
        iLine.set_linestyle(iLinestyle)
    
    eval(f"ax{iIteration}.legend([line for line in ax{iIteration}.lines], ['Basis', 'Lockdown', 'Medicine', 'Vaccine'])")

fig.tight_layout(pad=1.5)
