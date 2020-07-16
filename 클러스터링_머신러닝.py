#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
labelencoder=preprocessing.LabelEncoder()
from IPython.core.interactiveshell import InteractiveShell
from sklearn.cluster import DBSCAN



'''
from sklearn.mixture import GaussianMixture
from sklearn.utils.testing import ignore.warnings
from sklearn.exceptions import ConvergeneWarning

model=GaussianMixture(n_components=2, init params='random',random_state=0,tol=1e-9,max_iter=10)
model.fit(X)

pi=model.predict_proba(X)
plt.scatter(X[:,0],X[:,1],s=50,, linewidth=1, edgecolors="b")

'''
##Show the all data not skip middle data
InteractiveShell.ast_node_interactivity = "all"

# Load the data
Country= pd.read_csv('C:\\Users\\Jaewook_Cha\\Desktop\\world-development-indicators (3)\\Country.csv', encoding='utf-8')
Indicators=pd.read_csv('C:\\Users\\Jaewook_Cha\\Desktop\\world-development-indicators (3)\\Indicators.csv', encoding='utf-8')

# drop the unusing data
Country=Country.drop('ShortName',axis=1)
Country=Country.drop('TableName',axis=1)
Country=Country.drop('LongName',axis=1)
Country=Country.drop('Alpha2Code',axis=1)
Country=Country.drop('CurrencyUnit',axis=1)
Country=Country.drop('SpecialNotes',axis=1)
Country=Country.drop('Wb2Code',axis=1)
Country=Country.drop('NationalAccountsBaseYear',axis=1)
Country=Country.drop('NationalAccountsReferenceYear',axis=1)
Country=Country.drop('SnaPriceValuation',axis=1)
Country=Country.drop('LendingCategory',axis=1)
Country=Country.drop('OtherGroups',axis=1)
Country=Country.drop('SystemOfNationalAccounts',axis=1)
Country=Country.drop('AlternativeConversionFactor',axis=1)
Country=Country.drop('PppSurveyYear',axis=1)
Country=Country.drop('BalanceOfPaymentsManualInUse',axis=1)
Country=Country.drop('ExternalDebtReportingStatus',axis=1)
Country=Country.drop('SystemOfTrade',axis=1)
Country=Country.drop('GovernmentAccountingConcept',axis=1)
Country=Country.drop('ImfDataDisseminationStandard',axis=1)
Country=Country.drop('LatestPopulationCensus',axis=1)
Country=Country.drop('LatestHouseholdSurvey',axis=1)
Country=Country.drop('SourceOfMostRecentIncomeAndExpenditureData',axis=1)
Country=Country.drop('VitalRegistrationComplete',axis=1)
Country=Country.drop('LatestAgriculturalCensus',axis=1)
Country=Country.drop('LatestIndustrialData',axis=1)
Country=Country.drop('LatestTradeData',axis=1)
Country=Country.drop('LatestWaterWithdrawalData',axis=1)

##drop the nan value
Country=Country.dropna()

# Labeling the categorical data to traslate the number value
Country['Country_Encoder']=labelencoder.fit_transform(Country.iloc[:,0])
Country['Region_Encoder']=labelencoder.fit_transform(Country.iloc[:,1])

# Drop the unusing data in Indicator dataframe
Indicators=Indicators.drop('IndicatorName',axis=1)
Indicators=Indicators.drop('CountryName',axis=1)

# Shrink the number and see the more easy
Indicators['Year']=Indicators['Year']-1960

# Cut the data It's too many dat to calculate the code
Indicators=Indicators[Indicators['Year']<20]
# Reset the index to calculate the dataframe
Indicators=Indicators.reset_index(drop=True)

## pick the data then R1,R3,R4,R5,R6 each data 
Indicators_R1=Indicators[Indicators['IndicatorCode']=='TM.VAL.MRCH.R1.ZS']
Indicators_R3=Indicators[Indicators['IndicatorCode']=='TM.VAL.MRCH.R3.ZS']
Indicators_R4=Indicators[Indicators['IndicatorCode']=='TM.VAL.MRCH.R4.ZS']
Indicators_R5=Indicators[Indicators['IndicatorCode']=='TM.VAL.MRCH.R5.ZS']
Indicators_R6=Indicators[Indicators['IndicatorCode']=='TM.VAL.MRCH.R6.ZS']

# delete the nan data
Indicators_R1=Indicators_R1.dropna(how='all')
Indicators_R3=Indicators_R3.dropna(how='all')
Indicators_R4=Indicators_R4.dropna(how='all')
Indicators_R5=Indicators_R5.dropna(how='all')
Indicators_R6=Indicators_R6.dropna(how='all')

#Make the Indicator_set data frame and merge the all separated data.
Indicators_set=pd.concat([Indicators_R1,Indicators_R3,Indicators_R4,Indicators_R5,Indicators_R6],ignore_index=True)

#Reset the data index again
Indicators_set=Indicators.reset_index(drop=True)
Country=Country.reset_index(drop=True)

# Translate the Country Encoder to number value
Indicators_set['Country_Encoder']=labelencoder.fit_transform(Indicators_set.iloc[:,0])

count=0
# Cut the data It's too hard to calculate the all data.
Indicators_set=Indicators_set[Indicators_set['Country_Encoder']<150]

#Make the temp dataframe
Indicator_Sum=pd.DataFrame()

#Make the temp dataframe and Indicator_Sum will contain it
data_2=pd.DataFrame({"CountryCode":range(Indicators_set['Country_Encoder'].max()*Indicators_set['Year'].max())})
Indicator_Sum=Indicator_Sum.append(data_2)

data_3=pd.DataFrame({"Country_Encoder":range(Indicators_set['Country_Encoder'].max()*Indicators_set['Year'].max())})
Indicator_Sum=Indicator_Sum.append(data_3)

data_5=pd.DataFrame({"Income_Group":range(Indicators_set['Country_Encoder'].max()*Indicators_set['Year'].max())})
Indicator_Sum=Indicator_Sum.append(data_5)

data_4=pd.DataFrame({"Region_Encoder":range(Indicators_set['Country_Encoder'].max()*Indicators_set['Year'].max())})
Indicator_Sum=Indicator_Sum.append(data_4)

data_1=pd.DataFrame({"Sum":range(Indicators_set['Country_Encoder'].max()*Indicators_set['Year'].max())})
Indicator_Sum=Indicator_Sum.append(data_1)

data_6=pd.DataFrame({"Year":range(Indicators_set['Country_Encoder'].max()*Indicators_set['Year'].max())})
Indicator_Sum=Indicator_Sum.append(data_6)

# Set the free value to fill the except the nan value
Indicator_Sum['Region_Encoder']=0
Indicator_Sum['Income_Group']='a'
Indicator_Sum['Year']=0

#Translate the CountryCode value to String value.
Indicator_Sum['CountryCode']=Indicator_Sum['CountryCode'].astype(str)
pd.set_option('display.max_colwidth',-1)

# Translate the data_2 free dataframe to String
data_2=data_2.astype(str)

##Select the data Using Indicators_set data frame and Indicators_set
##To merge two dataframe
for i in range(Indicators_set['Country_Encoder'].max()):
    for j in range(Indicators_set['Year'].max()):
        if(Indicators_set[(Indicators_set['Country_Encoder']==i)&(Indicators_set['Year']==j)].empty==False):
            temp=Indicators_set[(Indicators_set['Country_Encoder']==i)&(Indicators_set['Year']==j)&(Indicators_set['IndicatorCode']=='TM.VAL.MRCH.R3.ZS')]
            q_1=temp.iloc[0:1,0]
            q_2=temp.iloc[0:1,4]
            temp_1=q_1.values.tolist()
            temp_2=q_2.values.tolist()
            
            if not temp_1:
                continue
            elif not temp_2:
                continue
            else:
                Indicator_Sum.iloc[count,0]=temp_1[0]
                Indicator_Sum.iloc[count,1]=int(temp_2[0])
                Indicator_Sum.iloc[count,4]=sum(Indicators_set['Value'][(Indicators_set['Country_Encoder']==i)&(Indicators_set['Year']==j)])
                count=count+1
#drop unusing data                
Indicator_Sum=Indicator_Sum.dropna(axis=0)
Indicator_Sum['Country_Encoder']=Indicator_Sum['Country_Encoder'].astype(int)

##Assing the Country data to merge the Indicator_Sum dataframe
for i in range(Indicator_Sum['Country_Encoder'].max()):
    for j in range(Country['Country_Encoder'].max()):
        if ((Indicator_Sum.iloc[i,0]) in (Country.iloc[j,0])):
            Indicator_Sum.iloc[i,2]=Country.iloc[j,2]
            Indicator_Sum.iloc[i,3]=Country.iloc[j,4]
            
# To seek the unusable dat set the nan value. and It will be droped.
for i in range(len(Indicator_Sum.iloc[:,1])):
    if Indicator_Sum.iloc[i,3]==0:
        Indicator_Sum.iloc[i,3]=np.nan

# Drop the Unusable data
Indicator_Sum.dropna(axis=0,inplace=True)
#Translate the categorical data to numbering
Indicator_Sum['Income_Group']=labelencoder.fit_transform(Indicator_Sum.iloc[:,2])
#Translate the float value to int value.
Indicator_Sum['Region_Encoder']=Indicator_Sum['Region_Encoder'].astype(int)
print(Indicator_Sum)

#여기까지 잘돈다 밑에 고치기#######################################################################3

'''
model=DBSCAN()
model.fit(Indicator_Sum.iloc[:,1:5])

y_predict=model.fit_predict
print(y_predict)
Indicator_Sum['cluster']=y_predict

print(Indicator_Sum)
'''



# In[ ]:





# In[ ]:




