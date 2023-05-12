import pandas as pd

database = pd.read_csv('C:\\Users\\santi\\OneDrive\\Desktop\\graphs\\EdStatsData.csv')
print(database.info())

World_Bank = pd.DataFrame(database)
database.count
                                   
#subsetting World Bank for Italy

Italy =  World_Bank[World_Bank['Country Code'] == 'ITA']

'''
Data preparation on italy subset:
    drop all rows  that cointan no data at all
    mantaining Indicator name and indicator code
    dropping redundant columns
'''


Italy_info_1 = Italy['Indicator Name']
Italy_info_2 = Italy['Indicator Code']

Italy = Italy.drop('Country Name',1).drop('Country Code',1).drop('Indicator Name',1).drop('Indicator Code',1)


#eliminating rows and columns with no data at all

Italy = Italy.dropna(how='all', axis=0)

#merging info and Italy database

Italy_kinda_ready = pd.merge(Italy_info_2, Italy, left_index=True, right_index=True)

Italy_ready_to_go = pd.merge(Italy_info_1, Italy_kinda_ready, left_index=True, right_index=True)


italy = Italy_ready_to_go.reset_index().drop('index',1)


'''
function to extract a row from italy to a dataframe with only one column
'''

def extract(x):
    variable = italy.loc[x]
    variable = variable.drop('Indicator Name').drop('Indicator Code')
    variable = pd.to_numeric(variable)
    return(variable)



'''
INSIGHT IV
'''

#looking for indicator names in italy subset


indicators = ['Percentage of students in lower secondary general education who are female (%)',
     'Percentage of teachers in lower secondary education who are female (%)',
     'PISA: Mean performance on the mathematics scale',
     'PISA: Mean performance on the reading scale',
     'PISA: Mean performance on the science scale',
     'Annual statutory teacher salaries in public institutions in USD. Primary. Starting salary',
     'Annual statutory teacher salaries in public institutions in USD. Upper Secondary. Starting salary',
     'Expenditure on education as % of total government expenditure (%)',
     'Government expenditure on education as % of GDP (%)'
]

#printing out indexes
for y in indicators:
 check_index = italy[italy['Indicator Name'] == y]
 print(check_index.index)

#using extract to create subsets (manual entry of indexes)

mat = extract(1138)
red = extract(1141)
sci = extract(1144)
female_student = extract(913)
female_teacher = extract(943)
salary_teach_prim = extract(42)
salary_teac_sec = extract(46)
expenditure_in_education_on_total_gov_expenditure = extract(554)
expenditure_as_percentage_of_GDP = extract(588)

#merging , renaming and dropna

for_insight_1 = pd.merge(expenditure_in_education_on_total_gov_expenditure, mat, left_index=True, right_index=True)
for_insight_1 = pd.merge(for_insight_1, red, left_index=True, right_index=True)
for_insight_1 = pd.merge(for_insight_1, sci, left_index=True, right_index=True)
for_insight_1 = pd.merge(for_insight_1, expenditure_as_percentage_of_GDP , left_index=True, right_index=True)

for_insight_1.rename(columns = {554 : 'education_on_total_expenditure', 
                                   1138 : 'mathematics',
                                   1141:'reading' ,
                                   1144:'science',
                                   588 : 'Edu_Expenditure_percentage_GDP'}, inplace = True)

for_insight_1 = for_insight_1.dropna(how='all', axis=0)


#correlation between performances and expenditure assuming expenditure has influence on performance of the same year

correlation_1 = for_insight_1[['education_on_total_expenditure',
                               'mathematics', 'reading', 'science',
                               'Edu_Expenditure_percentage_GDP' ]].corr()

correlation_1 = correlation_1.drop(['mathematics',
                                    'reading', 
                                    'science'], axis=1).drop(['education_on_total_expenditure',
                                                              'Edu_Expenditure_percentage_GDP'], axis=0)

#orrelation between performances and expenditure assuming 1 year for implementing expenditure

for_insight_1b = for_insight_1
for_insight_1b['education_on_total_expenditure'] = for_insight_1b.education_on_total_expenditure.shift(1)

correlation_1b = for_insight_1b[['education_on_total_expenditure',
                                 'mathematics', 'reading', 'science', 
                                 'Edu_Expenditure_percentage_GDP' ]].corr()

correlation_1b = correlation_1b.drop(['mathematics', 'reading','science'], axis=1).drop(['education_on_total_expenditure',
                                                                                         'Edu_Expenditure_percentage_GDP'], axis=0)


#merging and renaming of more parameters

for_insight_1c = pd.merge(salary_teach_prim, salary_teac_sec, left_index=True, right_index=True)
for_insight_1c = pd.merge(for_insight_1c, female_student, left_index=True, right_index=True)
for_insight_1c = pd.merge(for_insight_1c, female_teacher, left_index=True, right_index=True)
for_insight_1c = pd.merge(for_insight_1c, for_insight_1, left_index=True, right_index=True)

for_insight_1c.rename(columns = {42 : 'Primary teachers salary',
                                   'Edu_Expenditure_percentage_GDP' : 'Education expenditure on total GDP (%)',
                                   913 : 'percentage female students',
                                   943 : 'percentage female teachers',
                                   46 : 'Secondary teachers Salary',
                                   'education_on_total_expenditure' : 'Education expenditure on total expenditure (%)'}, inplace = True)


#correlation matrix 

correlation_1c = for_insight_1c.corr()
correlation_1c = correlation_1c.drop(['mathematics', 'reading','science'], axis=1).drop(['Primary teachers salary',
                                                                                         'Secondary teachers Salary',
                                                                                          'Education expenditure on total expenditure (%)',
                                                                                          'Secondary teachers Salary',
                                                                                          'Education expenditure on total GDP (%)',
                                                                                          'percentage female students',
                                                                                          'percentage female teachers'
                                                                                            ], axis=0)

#correlation matrix graph 

import seaborn as sns
import matplotlib.pyplot as plt


#aesthetic
sns.set_style("darkgrid")
sns.set_context("paper", rc={"lines.linewidth": 2})

cmap = sns.color_palette("vlag", as_cmap=True)
a = sns.heatmap(correlation_1c, cmap=cmap, vmax=1, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .6}, linecolor = 'white')
plt.title('Correlations in Italian lower secondary school', fontsize = 18)
plt.ylabel('PISA performances', fontsize = 15,labelpad=10)
a.set_xticklabels(a.get_xticklabels(), rotation=25, horizontalalignment='right')
a.set_yticklabels(a.get_yticklabels(), rotation=0, horizontalalignment='right')
cbar = a.collections[0].colorbar
cbar.set_ticks([0.9, 0, -.9])
cbar.set_ticklabels(['positive', 'null (0)', 'negative'])
plt.show()



'''
Some graph for introduction
'''


#graph on Italy performances over years

Pisa =  for_insight_1
Pisa = Pisa.drop(['education_on_total_expenditure', 'Edu_Expenditure_percentage_GDP'], axis=1)
Pisa = Pisa.dropna()


PISA = sns.lineplot(data=Pisa, markers=["o", "o","o"])
PISA.set_title("Italian performance over time (PISA)", fontsize = 20)
PISA.set_ylabel('mean PISA points')
plt.show()



Pisa["Pisa 2015 mean total"] = Pisa.sum(axis=1)/3

#merging and creating dataframes

uno = World_Bank[World_Bank['Indicator Name'] =='PISA: Mean performance on the mathematics scale']
dos = World_Bank[World_Bank['Indicator Name'] =='PISA: Mean performance on the reading scale']
tres = World_Bank[World_Bank['Indicator Name'] =='PISA: Mean performance on the science scale']


World_Pisa = pd.concat([uno,dos])
World_Pisa = pd.concat([World_Pisa,tres])


World_Pisa = World_Pisa.drop(['Country Name','Country Code','Indicator Name','Indicator Code'], axis=1)

World_Pisa = World_Pisa.dropna(how='all', axis=0)



'''
scatter plot for GDP and science performance 
'''
#creating merging and renaming dataframes 
gdp = World_Bank[World_Bank['Indicator Name'] =='GDP per capita, PPP (current international $)']
gdp = gdp.set_index('Country Name')
gdp_to_use = gdp['2015']

tres1 = tres.drop(['Indicator Name','Indicator Code', 'Country Code'], axis=1)
tres1 = tres1.set_index('Country Name')


sci_gdp = pd.merge(tres1,gdp, left_index=True, right_index=True)
sci_gdp.rename(columns = {'2015_x' : 'science performance','2015_y' : 'GDP per capita ($)'}, inplace = True)


#plot
sci_gdp.plot(x='GDP per capita ($)', y='science performance',
             linestyle='none', marker="o", markersize=10, markeredgecolor="orange", markeredgewidth=1,alpha=.9)
plt.yticks([350, 400, 450,500,550,600], ['350', '400', '450','500','550','600'])
plt.ylabel('Average PISA score',fontsize = 12,labelpad=10)
plt.title('PISA performance in science and GDP per capita',fontsize = 14)
plt.show()


#same plot for reading

#creating merging and renaming dataframes 
dos1 = dos.drop(['Indicator Name','Indicator Code', 'Country Code'], axis=1)
dos1 = dos1.set_index('Country Name')


read_gdp = pd.merge(dos1,gdp, left_index=True, right_index=True)
read_gdp.rename(columns = {'2015_x' : 'reading performance','2015_y' : 'GDP per capita ($)'}, inplace = True)

#plot
read_gdp.plot(x='GDP per capita ($)', y='reading performance',
             linestyle='none', marker="D", markersize=10, markeredgecolor="yellow", markeredgewidth=1,alpha=.9)
plt.yticks([350, 400, 450,500,550,600], ['350', '400', '450','500','550','600'])
plt.ylabel('Average PISA score',fontsize = 12,labelpad=10)
plt.title('PISA performance in reading and GDP per capita',fontsize = 14)
plt.show()

#same plot for math

#creating merging and renaming dataframes 
uno1 = uno.drop(['Indicator Name','Indicator Code', 'Country Code'], axis=1)
uno1 = uno1.set_index('Country Name')


read_gdp = pd.merge(uno1,gdp, left_index=True, right_index=True)
read_gdp.rename(columns = {'2015_x' : 'mathematics performance','2015_y' : 'GDP per capita ($)'}, inplace = True)

#plot
read_gdp.plot(x='GDP per capita ($)', y='mathematics performance',
             linestyle='none', marker="h", markersize=10, markeredgecolor="red", markeredgewidth=1,alpha=.9)
plt.ylabel('Average PISA score',fontsize = 12,labelpad=10)
plt.yticks([350, 400, 450,500,550,600], ['350', '400', '450','500','550','600'])
plt.title('PISA performance in mathematics and GDP per capita',fontsize = 14)
plt.show()





'''
graph for overall Pisa performance (2015) on GNI with marker sized by population
'''


#creating merging and renaming dataframes 

mat_2015 = uno1['2015']
read_2015 = dos1['2015']
sci_2015 = tres1['2015']

overall_15 = pd.merge(mat_2015, read_2015, left_index=True, right_index=True)
overall_15 = pd.merge(overall_15, sci_2015, left_index=True, right_index=True)
overall_15 = overall_15.dropna(how='all', axis=0)
overall_15["Pisa 2015"] = overall_15.sum(axis=1)/3
overall_15 = overall_15.drop(['2015', '2015_x','2015_y'], axis=1)


gni = World_Bank[World_Bank['Indicator Name'] =='GNI per capita, Atlas method (current US$)']
gni = gni.set_index('Country Name')
gni = gni['2015']


pop = World_Bank[World_Bank['Indicator Name'] =='Population, ages 10-15, total']
pop = pop.set_index('Country Name')
pop = pop['2015']

overall_gni = pd.merge(overall_15, gni, left_index=True, right_index=True)
overall_gni = pd.merge(overall_gni, pop, left_index=True, right_index=True)

overall_gni.rename(columns = {'Pisa 2015' : 'mean PISA performance',
                                   '2015_x' : 'GNI per capita ($)',
                                   '2015_y' : 'Population (billions)'
                                   }, inplace = True)


#plot
popg = sns.scatterplot(data=overall_gni, x='GNI per capita ($)', y='mean PISA performance', size='Population (billions)',legend = False,
                color='red', sizes=(10,5000), edgecolor="black", alpha=.7).set(title='PISA scores vs GNI index sized by 10-15 age population (2015)')
plt.show()



'''
INSIGHT I

one tail t-test for world on Pisa performances 2015 vs 2000



GNI per capita rates for 2015

https://blogs.worldbank.org/opendata/new-country-classifications-2016

New World Bank country classifications by income level:

lower middle-income  $1,026 and $4,035;
upper middle-income  $4,036 - $12,475;
high-income economies  > 12,476 

'''

from scipy.stats import ttest_rel


World_Pisa_t = World_Pisa['2000']
World_Pisa_t1 = World_Pisa['2015']
World_Pisa_t = pd.merge(World_Pisa_t, World_Pisa_t1, left_index=True, right_index=True).dropna()

print(World_Pisa_t.mean())


# t-test with significance level of 0.05
t_value, p_value = ttest_rel(World_Pisa_t['2000'], World_Pisa_t['2015'], alternative="greater")

print("p value 'greater' : " + str(p_value))
#p value = 0.97


#dropna
World_Pisa_v = World_Pisa.dropna(how='all', axis=1)
World_Pisa_v = World_Pisa_v.dropna()



#violin plot for each year 

plt.violinplot(World_Pisa_v[['2000', '2003', '2006','2009','2012','2015']], showmeans=True, showmedians=True,)
plt.title("World performances")
plt.xticks([1, 2, 3,4,5,6], ['2000', '2003', '2006','2009','2012','2015'])
plt.yticks([310,330,350,370,390,410,430,450,470,490,510,530,550,570])
plt.ylabel('TOTAL PISA POINTS (mean)')
plt.show()


print(overall_gni['GNI per capita ($)'].min()) #our dataset for 2015 does not have PISA values >1026 GNI  :(




#creating 2 different datasets for middle and high income countries
mid_gni_2015 = overall_gni[overall_gni['GNI per capita ($)']<12475]
mid_gni_2015.columns = ['PISA 2015', 'gni', 'pop']

high_gni_2015 = overall_gni[overall_gni['GNI per capita ($)']>12475]
high_gni_2015.columns = ['PISA 2015', 'gni', 'pop']

#creting dataset for mean score on 2000 PISA tests
mat_2000 = uno1['2000']
read_2000 = dos1['2000']
sci_2000 = tres1['2000']

#merging renaming and subsetting for GNI
overall_00 = pd.merge(mat_2000, read_2000, left_index=True, right_index=True)
overall_00 = pd.merge(overall_00, sci_2000, left_index=True, right_index=True)
overall_00 = overall_00.dropna(how='all', axis=0)
overall_00["Pisa 2000"] = overall_00.sum(axis=1)/3
overall_00 = overall_00.drop(['2000', '2000_x','2000_y'], axis=1)


overall_gni_0 = pd.merge(overall_00, gni, left_index=True, right_index=True)

mid_gni_2000 = overall_gni_0[overall_gni_0['2015']<12475]
mid_gni_2000.columns = ['PISA 2000', 'gni']

high_gni_2000 = overall_gni_0[overall_gni_0['2015']>12475]
high_gni_2000.columns = ['PISA 2000', 'gni']



mid_gni = pd.merge(mid_gni_2000, mid_gni_2015, left_index=True, right_index=True)

high_gni = pd.merge(high_gni_2000, high_gni_2015, left_index=True, right_index=True)

'''
overall Pisa (2000 and 2015)  divided in 2 portion  'middle' , 'high'
'''

print('mid 2000 = ' +  str(mid_gni['PISA 2000'].mean()))
print('mid 2015 = ' + str(mid_gni['PISA 2015'].mean()))

print('high 2000 = ' + str(high_gni['PISA 2000'].mean()))
print('high 2015 = ' + str(high_gni['PISA 2015'].mean()))


#t-test paired for middle income countries with level of significance of 0.05

t_value, p_value = ttest_rel(mid_gni['PISA 2000'], mid_gni['PISA 2015'])                              
print("p value for 2015 equal than 2000 (middle income) : " + str(p_value))                          #p value 2015 equal 2000 =  0.05
t_value, p_value = ttest_rel(mid_gni['PISA 2000'], mid_gni['PISA 2015'], alternative="less")
print("p value for 2015 less than 2000 (middle income) : " + str(p_value))                           #p value 2015 less than 2000 =  0.02
t_value, p_value = ttest_rel(mid_gni['PISA 2000'], mid_gni['PISA 2015'], alternative="greater")
print("p value for 2015 greater than 2000 (middle income) : " + str(p_value))                        #p value 2015 greater than 2000 =  0.97



#t-test paired for high income countries with level of significance of 0.05
t_value, p_value = ttest_rel(high_gni['PISA 2000'], high_gni['PISA 2015'],alternative="greater")
print("p value for 2015 greater to 2000 (high income) : " + str(p_value))                               #p value 2015 greater than 2000 =  0.45
t_value, p_value = ttest_rel(high_gni['PISA 2000'], high_gni['PISA 2015'], alternative="less")
print("p value for 2015 less to 2000 (high income) : " + str(p_value))                                  #p value 2015 less than 2000 = 0.54
t_value, p_value = ttest_rel(high_gni['PISA 2000'], high_gni['PISA 2015'])
print("p value for 2015 equal to 2000 (high income) : " + str(p_value))                                 #p value 2015 equal 2000 =  0.90


#violin plot for 2000 and 2015 scores grouped by mid and high income countries

plt.violinplot(mid_gni[['PISA 2000', 'PISA 2015']], showmeans=True, showmedians=False)
plt.violinplot(high_gni[['PISA 2000', 'PISA 2015']], showmeans=True, showmedians=False)
plt.title(('orange= high income countries     blue = middle income countries'))
plt.xticks([1, 2], ['2000','2015'])
plt.yticks([320,370,400,420,440,470,500,530,550])
plt.ylabel('POINTS (mean)')
plt.suptitle('PISA tests results over time grouped by income level')
plt.show()

'''
INSIGHT II
'''

#linear regression  between number of graduates 27 years before PISA test grouped by sex in high income countries

#merging, renaming , subsetting
fem = World_Bank[World_Bank['Indicator Code'] == 'UIS.EA.6.AG25T99.F']
fem = fem.set_index('Country Name')
print(fem.info(verbose=True, null_counts=True)) #from 2007 to 2012 are the year with most entries
fem = fem['2009']

mal = World_Bank[World_Bank['Indicator Code'] =='UIS.EA.6.AG25T99.M']
mal = mal.set_index('Country Name') 
print(mal.info(verbose=True, null_counts=True)) #from 2007 to 2012 are the year with most entries
mal = mal['2009']

mat_2010 = uno1['2009']
read_2010 = dos1['2009']
sci_2010 = tres1['2009']

#merging renaming and subsetting for GNI
overall_10 = pd.merge(mat_2010, read_2010, left_index=True, right_index=True)
overall_10 = pd.merge(overall_10, sci_2010, left_index=True, right_index=True)
overall_10 = overall_10.dropna(how='all', axis=0)
overall_10["Pisa 2009"] = overall_10.sum(axis=1)/3
overall_10 = overall_10.drop(['2009', '2009_x','2009_y'], axis=1)

mal = pd.merge(gni,mal,left_index=True, right_index=True)
mal = pd.merge(mal,overall_10,left_index=True, right_index=True)

fem = pd.merge(gni,fem,left_index=True, right_index=True)
fem = pd.merge(fem,overall_10,left_index=True, right_index=True)
fem.rename(columns = {
                                   '2009' : "percentage of female population 25+ with Bachelor's",
                                   }, inplace = True)

mal.rename(columns = {
                                   '2009' : "percentage of male population 25+ with Bachelor's",
                                   }, inplace = True)
mal = mal[mal['2015']>12475]
fem = fem[fem['2015']>12475]

fem = fem.dropna()
mal = mal.dropna()



from scipy import stats

#linear regression
slope, intercept, r_value, p_value, std_error = stats.linregress(mal["percentage of male population 25+ with Bachelor's"], mal['Pisa 2009'])
print("Slope male: " + str(slope))

#linear regression
slope, intercept, r_value, p_value, std_error = stats.linregress(fem["percentage of female population 25+ with Bachelor's"], fem['Pisa 2009'])
print("Slope female: " + str(slope))



#https://stackoverflow.com/questions/36026149/how-to-plot-multiple-linear-regressions-in-the-same-figure
fig, ax = plt.subplots(figsize=(6, 6))


#regplot divided by sex

sns.regplot(x="percentage of male population 25+ with Bachelor's", y='Pisa 2009', data=mal, fit_reg=True, ci=None, ax=ax, label='male')
sns.regplot(x="percentage of female population 25+ with Bachelor's", y='Pisa 2009', data=fem, fit_reg=True, ci=None, ax=ax, label='female').set(title='Parents Graduate population vs. childs PISA scores in high income countries')
ax.set(ylabel='PISA performance (2009)', xlabel="Percentage of population (25+) with Bachelor\'s degree or equivalent in 2009" )
ax.legend()
plt.show()


'''
INSIGHT III
'''

prim_age = World_Bank[World_Bank['Indicator Name'] =='Official entrance age to compulsory education (years)']

prim_age = prim_age.drop('Country Code',1).drop('Indicator Name',1).drop('Indicator Code',1)
prim_age = prim_age.set_index('Country Name')

prim_age_years = prim_age[['2009']]
prim_age_years['Entrance'] = prim_age_years.mean(axis=1)

#reusing a old variable with 2009 mean scores for PISA test
PISA_mean_2009 = overall_10

prim_y_Pisa = pd.merge(prim_age_years['Entrance'],PISA_mean_2009,left_index=True, right_index=True)


prim_y_Pisa = pd.merge(prim_y_Pisa['Entrance'],PISA_mean_2009,left_index=True, right_index=True)
prim_y_Pisa = pd.merge(prim_y_Pisa, gni,left_index=True, right_index=True)

prim_y_Pisa.loc[prim_y_Pisa["2015"] <= 12475, "2015"] = 'middle income'

prim_mid = prim_y_Pisa[prim_y_Pisa['2015'] == 'middle income']
prim_y_Pisa = prim_y_Pisa[prim_y_Pisa['2015'] != 'middle income']

prim_y_Pisa.loc[prim_y_Pisa["2015"] >= 12475, "2015"] = 'high income'

prim_all = pd.concat([prim_y_Pisa,prim_mid])

this = [prim_all, prim_mid]

years_primary = pd.concat(this)


last_plot = sns.boxplot( x='Entrance', y='Pisa 2009', data=years_primary,width=0.9, hue="2015")
last_plot.legend(title="Countries")
last_plot.set_title('Primary school starting age vs. Pisa scores (2009)')
last_plot.set_xlabel("Primary school starting age", fontsize = 14)
last_plot.set_ylabel("Pisa average scores (2009)", fontsize = 11)
plt.show()

four_high = prim_y_Pisa[prim_y_Pisa['Entrance'] == 4 ].drop('Entrance',1).drop('2015',1)
five_high = prim_y_Pisa[prim_y_Pisa['Entrance'] == 5 ].drop('Entrance',1).drop('2015',1)
six_high = prim_y_Pisa[prim_y_Pisa['Entrance'] == 6 ].drop('Entrance',1).drop('2015',1)
seven_high = prim_y_Pisa[prim_y_Pisa['Entrance'] == 7 ].drop('Entrance',1).drop('2015',1)

seven = prim_mid[prim_mid['Entrance'] == 7 ].drop('Entrance',1).drop('2015',1)
six = prim_mid[prim_mid['Entrance'] == 6 ].drop('Entrance',1).drop('2015',1)
five = prim_mid[prim_mid['Entrance'] == 5 ].drop('Entrance',1).drop('2015',1)
four = prim_mid[prim_mid['Entrance'] == 4 ].drop('Entrance',1).drop('2015',1)

from scipy.stats import f_oneway


#ANOVA for starting age on high income countries      0.05 level of significance

f_value, p_value = f_oneway(seven_high,six_high,five_high,four_high)
print("F-score: " + str(f_value))
print("p value ANOVA high: " + str(p_value))

  

#ANOVA for starting age on mid income countries    0.05 level of significance

f_value, p_value = f_oneway(seven,six,five,four)
print("F-score: " + str(f_value))
print("p value ANOVA mid: " + str(p_value))





















