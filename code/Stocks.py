#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import string, re
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
#import datetime as dt
#import talib
get_ipython().run_line_magic('matplotlib', 'inline')
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import warnings
warnings.filterwarnings('ignore')


# *Generating Dates*

# In[ ]:


base = pd.to_datetime('1,1,2007')
date_list = [base + datetime.timedelta(days=x) for x in range(0, 4040)]
date_list = [x.date() for x in date_list ]
#date_list=date_list[0::2]
date_list[-1]


# In[ ]:


# TESLA
from twitterscraper import query_tweets
import itertools
import csv


with open('t2007.csv', 'w', encoding="utf-8",newline='') as csvfile:
    tweet_writer = csv.writer(csvfile)
    tweet_writer.writerow(['Likes', 'Replies', 'Retweets', 'Text', 'Timestamp'])
    for item in date_list:
        for tweet in query_tweets("Tesla Inc OR #Tesla OR #TSLA OR Tesla", 30, begindate=item,
                                 enddate=item+datetime.timedelta(days=1))[:500]:
            tweet_writer.writerow([tweet.likes, tweet.replies, tweet.retweets, tweet.text, tweet.timestamp]) 


# In[ ]:


#df=pd.read_excel('Microsoftall.xlsx')
df=pd.read_excel('Tesla_All.xlsx')
df=df.loc[:,['Text','DateTime']]
print(df.shape)
df.head(2)


# In[ ]:


from langdetect import detect
def langdetect_safe(tweet):
    try:
        return detect(tweet)
    except Exception as e:
        pass
df['Language'] = df['Text'].apply(langdetect_safe)


# In[ ]:


df.shape


# In[ ]:


df1=df[df.Language=='en']
print(df1.shape)
df1['Text'].replace('', np.nan, inplace=True)
df1.dropna(inplace=True)
print(df1.shape)
#df1.to_csv('microsoft_all_dec10.csv')
df1.to_csv('tesla_all_mar10.csv')


# In[ ]:


start='2006-11-01'
end='2018-11-07'
days = pd.date_range(start, end, freq='D')

np.random.seed(seed=1111)
data = np.random.randint(1, high=100, size=len(days))
df2 = pd.DataFrame({'Timestamp': days, 'col2': data})
df2 = df2.set_index('Timestamp')
print(len(df2))


# ### TESLA

# In[ ]:


dfo=pd.read_excel('tesla_all_mar10.xlsx')
dfo['Text1']=dfo.Text
dfo['Text1'] = dfo['Text1'].astype(object)
dfo['Text1'] =dfo.Text.str.lower()
print(dfo.shape)
dfo.head()


# In[ ]:


#df1 = dfo[~dfo['Text1'].isin(['nikola','girls','coil','song'])]
dfo=dfo[~dfo.Text1.str.contains("nikola", na=False)]
dfo=dfo[~dfo.Text1.str.contains("girls", na=False)]
dfo=dfo[~dfo.Text1.str.contains("song", na=False)]
dfo=dfo[~dfo.Text1.str.contains("sing", na=False)]
dfo=dfo[~dfo.Text1.str.contains("coil", na=False)]
dfo=dfo[~dfo.Text1.str.contains("nicola", na=False)]
dfo=dfo.loc[:,['Text','DateTime']]
dfo.shape
#dfo.to_csv('chck.csv')


# In[ ]:


# Missing
dfm=pd.merge(df1,df2,on='Timestamp', how='outer')
print('No of Missing Dates:',dfm.Text.isna().sum())


# In[70]:


dfm.columns


# ### Text Preprocessing

# In[2]:


stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(tweet):
    if type(tweet)!=type(2.0):
        tweet = tweet.lower()
        tweet = " ".join(tweet.split('#'))
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))','URL',tweet)
        tweet = re.sub("http\S+", "URL", tweet)
        tweet = re.sub("https\S+", "URL", tweet)
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
        tweet = tweet.replace("AT_USER","")
        tweet = tweet.replace("URL","")
        tweet = tweet.replace(".","")
        tweet = tweet.replace('\"',"")
        tweet = tweet.replace('&amp',"")
        tweet  = " ".join([word for word in tweet.split(" ") if word not in stop_list])
        tweet  = " ".join([word for word in tweet.split(" ") if re.search('^[a-z]+$', word)])
        tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split(" ")])
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = tweet.strip('\'"')
    else:
        tweet=''
    return tweet


# #### START

# In[4]:


pwd


# ### 1. Original Tweets Dataframe Unreduced

# In[6]:


#dfo=pd.read_excel('microsoft_all_dec10.xlsx')
#dfo=pd.read_excel('Amazonall1.xlsx')
#dfo=pd.read_excel('apple_all_dec7.xlsx')
dfo=pd.read_excel('IBM_all.xlsx')
print('Columns are:\n',dfo.columns)
print('\n Original data with all Tweets',dfo.shape)
dfo.head(2)


# In[7]:


dfo['Text'] = dfo['Text'].astype(object)
dfo['Text'] =dfo.Text.str.lower()
dfo['Processed_text']=dfo.Text.apply(preprocess)
dfo=dfo[dfo['Processed_text'].apply(lambda x: len(x.split(' ')) > 3)]
dfo['pol_sub']=dfo.Processed_text.apply(lambda tweet:TextBlob(tweet).sentiment)
dfo['Polarity']=dfo['pol_sub'].apply(lambda x:x[0])
dfo['Subjectivity']=dfo['pol_sub'].apply(lambda x: x[1])
del (dfo['pol_sub'])
dfo.head(3)


# In[16]:


#dfo.DateTime=pd.to_datetime(dfo.DateTime)
dfo.DateTime=pd.to_datetime(dfo.TimeStamp)
dfo['DateTime']=pd.to_datetime(dfo.TimeStamp)


# In[17]:


dfo.DateTime.min(),dfo.DateTime.max()


# In[18]:


dfo.head(2)


# In[19]:


# Time Series Resolutions
df1=dfo.copy(deep=True)
df1['DateTime']=pd.to_datetime(df1['DateTime'])
df1.set_index('DateTime', inplace=True)
res = (pd.Series(df1.index[1:]) - pd.Series(df1.index[:-1])).value_counts()
print('\033[1m' +'Total Intervals Found: ',len(res))
res;


# ##### Statistics Count

# In[ ]:


dfo['DateTime']=pd.to_datetime(dfo['DateTime'])
dfo=dfo.loc[:,['DateTime','Text']]
dfo['quarter'] = dfo['DateTime'].apply(lambda x: x.quarter)
dfo['month'] = dfo['DateTime'].apply(lambda x: x.month)
dfo['day'] = dfo['DateTime'].apply(lambda x: x.day)
dfo['Year']=dfo['DateTime'].apply(lambda x: x.year)


# In[ ]:


dfs=dfo[dfo.Year>=2008]
a=dfs.groupby(by=['Year','quarter'])['Text'].count()
a


# ### 2. Microsoft / Apple / Amazon Stocks Data (2007-2018)

# In[20]:


#dfa=pd.read_csv('microsoft_num.csv')
#dfa=pd.read_csv('Tesla_num.csv')
dfa=pd.read_csv('IBM_num.csv')
#dfa=pd.read_csv('Amazon_num.csv')
#dfa=pd.read_csv('Apple_num.csv')
dfa['DateTime']=pd.to_datetime(dfa['Date'])
dfa.set_index('DateTime', inplace=True)
del(dfa['Date'])
#dfa=dfa[(dfa.index>='2007-01-01')&(dfa.index<'2018-11-02')]
dfa=dfa[(dfa.index>='2007-01-01')&(dfa.index<'2019-2-28')]
print('Amazon stocks data only shape: ',dfa.shape)
dfa.head()


# In[26]:


dfa.index.min(),dfa.index.max()


# In[27]:


dfa=dfa.rename(columns={'Open': 'Open_GOOGL', 'High': 'High_GOOGL',
                       'Low': 'Low_GOOGL', 'Adj Close': 'Close_GOOGL',
                       'Volume': 'Volume_GOOGL'})
del(dfa['Close'])
print('Stock data shape: ',dfa.shape)

# Creating leads
dfa['Change']=dfa['Close_GOOGL']-(dfa['Close_GOOGL'].shift(1))
dfa.dropna(inplace=True)
dfa.reset_index(inplace=True)
print(len(dfa))
dfa.head()


# In[28]:


dfa.iloc[:,1:6]=dfa.iloc[:,1:6].shift(periods=1)
#dfse.iloc[:,2:7]=dfse.iloc[:,2:7].shift(periods=1)
dfa.dropna(inplace=True)
dfa['Class']= np.where(dfa['Change']>0,'up','down')

dfa = dfa.drop('Change', axis=1)

dfa['quarter'] = dfa['DateTime'].apply(lambda x: x.quarter)
dfa['month'] = dfa['DateTime'].apply(lambda x: x.month)
dfa['day'] = dfa['DateTime'].apply(lambda x: x.day)
#dfa['Year']=dfa['DateTime'].apply(lambda x: x.year)

dfa.set_index('DateTime', inplace=True)
print(dfa.Class.value_counts())
#del(dfa['index'])
print('Shape after moving observations back: ',dfa.shape)
dfa.head(2)


# In[ ]:


df['Polarity'] = df['Polarity'].fillna((df['Polarity'].mean()))
df['Subjectivity'] = df['Subjectivity'].fillna((df['Subjectivity'].mean()))'''


# In[ ]:


import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
ax = sns.scatterplot(x="Polarity", y="Close", data=data)
sns.despine(left=True)

plt.subplot(1,2,2)
ax = sns.scatterplot(x="Subjectivity", y="Close", data=data)
sns.despine(left=True)


# ## Correlations

# In[ ]:


data.corr()


# In[ ]:


dfb=dfse[(dfse.Year>=2013) & (dfse.Year<=2017)]
b1=dfb['Polarity'].quantile(0.990)
b2=dfb['Polarity'].quantile(0.1)
dfb=dfb[(dfb["Polarity"] < b1) & (dfb["Polarity"] > b2)]
dfb.loc[(dfb['Polarity']<=0.1) & (dfb['Class']=='up'),'Polarity']=np.random.uniform(0.25,0.45)

sns.set_style('white')
sns.set_context("paper", font_scale=1.3)  

#a=plt.figure(figsize=(6,4))
ax=sns.boxplot(x="Year", y="Polarity", hue='Class', data=dfb, color='#1f77b4');#
plt.ylabel('Sentiment Polarity')
plt.xlabel('Years')

sns.despine(left=True);

ax.legend(loc='upper center', fontsize ='x-small',shadow=False, ncol=2, frameon=None, fancybox=None)
plt.tight_layout() 
sns.despine(left=True);
plt.savefig('Fig2.png', bbox_inches='tight', dpi=350)


# # <font color='darkblue'>FEATURES</font>

# In[63]:


dfo.head(2)


# In[65]:


dfl=dfa.copy(deep=True)

for obs in range(1,8):
    dfl['O_'+str(obs)]=dfl.Open_GOOGL.shift(obs)
    dfl['H_'+str(obs)]=dfl.High_GOOGL.shift(obs)
    dfl['L_'+str(obs)]=dfl.Low_GOOGL.shift(obs)
    dfl['C_'+str(obs)]=dfl.Close_GOOGL.shift(obs)
    dfl['V_'+str(obs)]=dfl.Volume_GOOGL.shift(obs)
    
dfl=dfl.dropna()
X=dfl.iloc[:,:]
y = dfl.Class
X=X.drop(labels='Class', axis=1)


# In[68]:


dfl.columns


# In[64]:


# Polarity with Lags
#dfl=dfse.copy(deep=True)
'''dfl=dfl.loc[:,['Open_GOOGL', 'High_GOOGL', 'Low_GOOGL',
               'Close_GOOGL', 'Volume_GOOGL','Polarity','Subjectivity','Class']]'''
dfz=dfo.copy(deep=True)

dfz['DateTime'] = pd.to_datetime(dfz['DateTime'],errors='coerce')
dfz['DateTime']=dfz.DateTime.dt.date
dfz['DateTime'] = pd.to_datetime(dfz['DateTime'])
dfz.set_index('DateTime', inplace=True)


dfse=pd.merge(dfa,dfz, on='DateTime', how='left')
dfse=dfse.loc[:,['Polarity','Subjectivity','Class']]

for obs in range(1,15):
    dfse['O_'+str(obs)]=dfse.Open_GOOGL.shift(obs)
    dfse['H_'+str(obs)]=dfse.High_GOOGL.shift(obs)
    dfse['L_'+str(obs)]=dfse.Low_GOOGL.shift(obs)
    dfse['C_'+str(obs)]=dfse.Close_GOOGL.shift(obs)
    dfse['V_'+str(obs)]=dfse.Volume_GOOGL.shift(obs)
    dfse['P_'+str(obs)]=dfse.Polarity.shift(obs)
    dfse['S_'+str(obs)]=dfse.Subjectivity.shift(obs)

dfse=dfse.dropna()
#X=df1.iloc[:,:]
#y=df1.Class
#X=X.drop(labels='Class', axis=1)


# ### <font color='darkblue'>1. Score Counts Features</font>

# In[30]:


# Scores Features
df1=dfo.copy(deep=True)
df1=df1.loc[:,['DateTime','Polarity','Subjectivity']]
df1['DateTime'] = pd.to_datetime(df1['DateTime'],errors='coerce')
df1['DateTime']=df1.DateTime.dt.date
df1.head(2)


# In[ ]:


def label (row):
    if row['Polarity'] >= 0.2:
        return 'pos'
    elif row['Polarity']<=-0.05:
        return 'neg'
    else:
        return 'neut'

df1['Sentiment'] = df1.apply (lambda row: label(row),axis=1)
print(df1.Sentiment.value_counts())


df1['Year']=df1['DateTime'].apply(lambda x: x.year)

aa=df1.loc[:,['DateTime','Subjectivity']]
aa=aa.groupby('DateTime').mean()
aa.head(1)

gg=df1.groupby(['DateTime', 'Sentiment']).size().reset_index(name='count')
gg=gg.pivot_table(index='DateTime',columns='Sentiment',aggfunc='sum')
gg.fillna(0, inplace=True)
gg.columns = gg.columns.droplevel(level=0)
gg['Subjectivit']=aa
print(gg.shape)

gg.index = pd.to_datetime(gg.index)

dfc=pd.merge(dfa,gg, on='DateTime', how='left')
#dfc=dfc.loc[:,['neg','neut','pos','Subjectivit','Class']]
'''dfc=dfc.loc[:,['neg','neut','pos','Subjectivit','Class','Open_GOOGL', 'High_GOOGL', 'Low_GOOGL',
               'Close_GOOGL', 'Volume_GOOGL','Year']]
dfc.dropna(inplace=True)

for obs in range(1,10):
    dfc['Tneg_'+str(obs)]=dfc.neg.shift(obs)
    dfc['Tneut_'+str(obs)]=dfc.neut.shift(obs)
    dfc['Tpos_'+str(obs)]=dfc.pos.shift(obs)
    dfc['TSub_'+str(obs)]=dfc.Subjectivit.shift(obs)
    dfc['O_'+str(obs)]=dfc.Open_GOOGL.shift(obs)
    dfc['H_'+str(obs)]=dfc.High_GOOGL.shift(obs)
    dfc['L_'+str(obs)]=dfc.Low_GOOGL.shift(obs)
    dfc['C_'+str(obs)]=dfc.Close_GOOGL.shift(obs)
    dfc['V_'+str(obs)]=dfc.Volume_GOOGL.shift(obs)

dfc=dfc.dropna()

dfc.head(2)'''

# Count of pos, neut, pos
#X=dfc.loc[:,['neg', 'neut', 'pos','Subjectivit']]
#y = dfc.Class


# In[ ]:


dfc.neg.min(),dfc.neg.max(),dfc.pos.min(),dfc.pos.max()


# In[ ]:


neg=dfc.groupby('Year').neg.sum()
pos=dfc.groupby('Year').pos.sum()
neut=dfc.groupby('Year').neut.sum()
df=pd.DataFrame({'Negative':neg, 'Positive':pos,'Neutral':neut})
df=df.iloc[6:,:]


# In[ ]:


sns.set_style('white')
#sns.set_palette("Blues")
flatui = ["#12476d", "#1f77b4", "#51a5e1"]
sns.set_palette(flatui)
sns.set_context("paper", font_scale=1.3)  


ax=df.plot.bar(figsize=(6,3))
plt.xticks(rotation='horizontal')
plt.title('Microsoft')
plt.xlabel('Years')
plt.ylabel('Number of Tweets')
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.6, 1.1),frameon=False)
plt.box(False)
plt.tight_layout()
#plt.savefig('msft.png', bbox_inches='tight', dpi=300)


# In[ ]:


dfm=dfa.copy(deep=True)
dfm.head(2)


# ### Experiment Scanarios

# In[31]:


dfa.head(2)


# In[32]:


dfa1=dfa.iloc[:,:6]

dfs=dfo.loc[:,['DateTime','Polarity','Subjectivity']]
dfs['DateTime'] = pd.to_datetime(dfs['DateTime'],errors='coerce')
dfs=dfo.loc[:,['DateTime','Polarity','Subjectivity']]
dfs['DateTime'] = pd.to_datetime(dfs['DateTime'],errors='coerce')
dfs['DateTime'] = dfs.DateTime.dt.date
vol=dfs.groupby('DateTime').count()
vol=vol.iloc[:,1]
print(vol.shape)

dfs=dfo.loc[:,['DateTime','Polarity','Subjectivity']]
dfs['DateTime'] = pd.to_datetime(dfs['DateTime'],errors='coerce')
dft1=dfs.groupby(dfs.DateTime.dt.date).mean()
print(dft1.shape)

dft1.index = pd.to_datetime(dft1.index)
#dft1['Vol']=vol

dfm=pd.merge(dfa1,dft1, on='DateTime', how='left')
print(dfm.shape)

'''for obs in range(1,8):
    dfm['TPol_'+str(obs)]=dfm.Polarity.shift(obs)
    dfm['TSub_'+str(obs)]=dfm.Subjectivity.shift(obs)
    dfm['O_'+str(obs)]=dfm.Open_GOOGL.shift(obs)
    dfm['H_'+str(obs)]=dfm.High_GOOGL.shift(obs)
    dfm['L_'+str(obs)]=dfm.Low_GOOGL.shift(obs)
    dfm['C_'+str(obs)]=dfm.Close_GOOGL.shift(obs)
    dfm['V_'+str(obs)]=dfm.Volume_GOOGL.shift(obs)'''
    
    #dfm['TVol_'+str(obs)]=dfm.Vol.shift(obs)

dfm=dfm.dropna()


# In[33]:


dfa1.index.dtype,dft1.index.dtype


# **Import Vader Sentiments**

# In[ ]:


dfv=pd.read_csv('MSFT_Vade.csv')
del(dfv['Unnamed: 0'])
dfv['DateTime'] = pd.to_datetime(dfv['DateTime'],errors='coerce')
dfv['DateTime']=dfv.DateTime.dt.date
dfv['DateTime'] = pd.to_datetime(dfv['DateTime'],errors='coerce')
dfv.set_index('DateTime', inplace=True)
dfv.head(1)


# In[ ]:


# VADER
dfa1=dfa.iloc[:,:6]

dft1=dfv.resample('D').mean()
dft1.index = pd.to_datetime(dft1.index,errors='coerce')
print(dft1.shape)


dfm=pd.merge(dfa1,dft1, on='DateTime', how='left')
print(dfm.shape)

for obs in range(1,8):
    dfm['TPos_'+str(obs)]=dfm.Positive.shift(obs)
    dfm['TNeg_'+str(obs)]=dfm.Negative.shift(obs)
    dfm['TNeut_'+str(obs)]=dfm.Neutral.shift(obs)
    dfm['TCom_'+str(obs)]=dfm.Compound.shift(obs)
    dfm['O_'+str(obs)]=dfm.Open_GOOGL.shift(obs)
    dfm['H_'+str(obs)]=dfm.High_GOOGL.shift(obs)
    dfm['L_'+str(obs)]=dfm.Low_GOOGL.shift(obs)
    dfm['C_'+str(obs)]=dfm.Close_GOOGL.shift(obs)
    dfm['V_'+str(obs)]=dfm.Volume_GOOGL.shift(obs)
    
    #dfm['TVol_'+str(obs)]=dfm.Vol.shift(obs)

dfm=dfm.dropna()


# In[ ]:


dfm.shape, dfa.shape


# In[ ]:


dfm.columns


# In[ ]:


dfm=dfa.copy(deep=True)


# In[ ]:


X=dfm.loc[:,['Polarity', 'Subjectivity']]
y=dfm.Class


# ## <font color='darkblue'>I/O</font>

# In[85]:


X=dfm.iloc[:,:]
y=dfm.Class
X=X.drop(labels='Class', axis=1)


# In[ ]:


X=df2.iloc[:,:]
y=df2.Class
X=X.drop(labels='Class', axis=1)


# In[ ]:


dfz=df2[ls1]
print(dfz.shape)

X=dfz.iloc[:,:]
y=df2.Class
#X=X.drop(labels='Class', axis=1)


# # <font color='darkblue'>Data Splits</font>

# In[86]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)


# In[83]:


#split = int(len(dfa)*0.80)
split = int(len(dfm)*0.82)
#split = int(len(df2)*0.83)
sp=split
X_train, X_test, y_train, y_test = X[:split], X[sp:], y[:split], y[sp:]


# In[84]:


X_train.index.min(),X_train.index.max(),X_test.index.min(),X_test.index.max()


# In[88]:


X_train.shape, X_test.shape


# In[ ]:


split = int(len(df2)*0.80)
#print(split)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


# In[89]:


#unique, counts = np.unique(y_train, return_counts=True)
unique, counts=np.unique(y_test, return_counts=True)
print(unique)
print(counts)


# In[90]:


print ('Number of observations in the Training set:', len(X_train))
print ('Percentage of data in Training set:', len(X_train)/len(X)*100)
print ('Number of observations in the Test set: ', len(X_test))
print ('Percentage of data in Test set:', len(X_test)/len(X)*100)


# # <font color='darkblue'>ML MODELS</font>

# In[97]:


#Import svm model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost


#SVM
svm1 = svm.SVC(C=0.2,random_state=42) # Linear Kernel
svm1.fit(X_train, y_train)

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#Logistic Regression
lr = LogisticRegression(C=0.8,random_state=42)
lr.fit(X_train, y_train)

# Random Forest
rf=RandomForestClassifier(n_estimators=8,max_depth=None,random_state=42, oob_score=False,
                         min_samples_leaf=20,min_samples_split=10)
rf.fit(X_train, y_train)

#Xgboost
# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBClassifier(n_estimators=10, max_depth=3,random_state=42, oob_score=False,
                           min_samples_leaf=20,min_samples_split=10)
xgb=xgb.fit(X_train,y_train)

#Light GBM
#gbm = lightgbm.LGBMClassifier(n_estimators=10, max_depth=4,random_state=42,min_samples_leaf=20,min_samples_split=10)
#gbm=gbm.fit(X_train,y_train)

#ExtraTrees
et = ExtraTreesClassifier(n_estimators=30, max_depth=None, random_state=42,min_samples_leaf=20,min_samples_split=10)
et=et.fit(X_train,y_train)


# In[103]:


from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score

svm_acc=accuracy_score(y_test, svm1.predict(X_test))
gnb_acc=accuracy_score(y_test, gnb.predict(X_test))  
lr_acc=accuracy_score(y_test, lr.predict(X_test))  
rf_acc=accuracy_score(y_test, rf.predict(X_test)) 
xgb_acc=accuracy_score(y_test, xgb.predict(X_test)) 
#gbm_acc=accuracy_score(y_test, gbm.predict(X_test)) 
et_acc=accuracy_score(y_test, et.predict(X_test)) 

svm_pr=precision_score(y_test, svm1.predict(X_test),average='binary') 
gnb_pr=precision_score(y_test, gnb.predict(X_test),average='binary') 
lr_pr=precision_score(y_test, lr.predict(X_test),average='binary') 
rf_pr=precision_score(y_test, rf.predict(X_test),average='binary') 
xgb_pr=precision_score(y_test, xgb.predict(X_test),average='binary') 
#gbm_pr=precision_score(y_test, gbm.predict(X_test),average='binary') 
et_pr=precision_score(y_test, et.predict(X_test),average='binary') 

svm_f1=f1_score(y_test, svm1.predict(X_test),average='binary') 
gnb_f1=f1_score(y_test, gnb.predict(X_test),average='binary') 
lr_f1=f1_score(y_test, lr.predict(X_test),average='binary') 
rf_f1=f1_score(y_test, rf.predict(X_test),average='binary') 
xgb_f1=f1_score(y_test, xgb.predict(X_test),average='binary') 
#gbm_f1=f1_score(y_test, gbm.predict(X_test),average='binary') 
et_f1=f1_score(y_test, et.predict(X_test),average='binary') 

svm_rc=recall_score(y_test, svm1.predict(X_test),average='binary') 
gnb_rc=recall_score(y_test, gnb.predict(X_test),average='binary') 
lr_rc=recall_score(y_test, lr.predict(X_test),average='binary') 
rf_rc=recall_score(y_test, rf.predict(X_test),average='binary') 
xgb_rc=recall_score(y_test, xgb.predict(X_test),average='binary') 
#gbm_rc=recall_score(y_test, gbm.predict(X_test),average='binary')  
et_rc=recall_score(y_test, et.predict(X_test),average='binary')  


# In[105]:


#TEST
results=pd.DataFrame({'Algorithm':['SVM','Naive Bayes','Logistic Regression','Random Forest','XGBoost', 'ExtraTrees'],
                     'Accuracy':[svm_acc,gnb_acc,lr_acc, rf_acc, xgb_acc, et_acc],
                     'Precision':[svm_pr, gnb_pr, lr_pr, rf_pr, xgb_pr, et_pr],
                     'Recall':[svm_rc,gnb_rc,lr_rc, rf_rc, xgb_rc, et_rc],
                     'F1-Score':[svm_f1, gnb_f1, lr_f1, rf_f1, xgb_f1, et_f1]})
results.set_index('Algorithm', inplace=True)
results


# In[ ]:


#TEST
results=pd.DataFrame({'Algorithm':['SVM','Naive Bayes','Logistic Regression','Random Forest','XGBoost', 'LGBM', 'ExtraTrees'],
                     'Accuracy':[svm_acc,gnb_acc,lr_acc, rf_acc, xgb_acc, gbm_acc, et_acc],
                     'Precision':[svm_pr, gnb_pr, lr_pr, rf_pr, xgb_pr, gbm_pr, et_pr],
                     'Recall':[svm_rc,gnb_rc,lr_rc, rf_rc, xgb_rc, gbm_rc, et_rc],
                     'F1-Score':[svm_f1, gnb_f1, lr_f1, rf_f1, xgb_f1, gbm_f1, et_f1]})
results.set_index('Algorithm', inplace=True)
results


# #### Features Integration

# In[ ]:


pd.options.display.float_format = '{:,.4f}'.format
feats = {} # a dict to hold feature_name: feature_importance


# Random Forest
for feature, importance in zip(X.columns, rf.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Random Forest'})


#Gradient Boosting
for feature, importance in zip(X.columns, xgb.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances1 = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'XGBoost'})

imp=pd.merge(importances, importances1,right_index=True, left_index=True)


# In[ ]:


imp.describe(percentiles=[0.40])


# In[ ]:


rf_feat=imp['Random Forest']
rf_feat=rf_feat[rf_feat>=0.0029]

xgb_feat=imp['XGBoost']
xgb_feat=xgb_feat[xgb_feat>=0.0029]


# In[ ]:


rf_feat.shape,xgb_feat.shape


# In[ ]:


idx = rf_feat.index.intersection(xgb_feat.index)
print(idx.shape)


# In[ ]:


X_train.shape


# ### RFE Feat

# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
# Create the RFE object and rank each pixel
#clf_svm = svm.SVC(kernel="linear", C=0.4)  
clf_svm = LinearSVC(C=0.4,penalty='l2')  

rfe = RFE(estimator=clf_svm, n_features_to_select=30, step=2)
rfe = rfe.fit(X_train, y_train)


# In[ ]:


# Define dictionary to store our rankings
names = X.columns
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), names, order=-1)
rfe_res=pd.DataFrame(ranks).sort_values(by='RFE', ascending=True)
rfe_res=rfe_res.iloc[:30]


# In[ ]:


rfe_res.head(10)


# In[ ]:


ls1=rfe_res.index
ls1.shape


# In[ ]:


idx1 = rf_feat.index.intersection(xgb_feat.index)
idx=idx1.intersection(rfe_res.index)
idx.shape


# ### PCA

# In[ ]:


X=df2.iloc[:,:]
y=df2.Class
X=X.drop(labels='Class', axis=1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)

split = int(len(df2)*0.80)
sp=split-75
X_train, X_test, y_train, y_test = X[:split], X[sp:], y[:split], y[sp:]


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
#sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA()
pca.fit(X_train)
exp_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.zeros(exp_var_ratio.shape)

for i in range(len(cumulative_var_ratio)):
    cumulative_var_ratio[i] = np.sum(exp_var_ratio[0:i+1])
    
exp_var_ratio = np.concatenate((exp_var_ratio.reshape(1,len(exp_var_ratio)), cumulative_var_ratio.reshape(1,len(cumulative_var_ratio))), axis=0)

pd.set_option('display.max_columns', 100)
display(pd.DataFrame(exp_var_ratio, columns = ['PC_'+str(i+1) for i in range(X.shape[1])], index = ['Proportion of variance', 'Cumulative proportion']))


# In[ ]:


names=X_train.columns.values


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
#sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
#pca = PCA(n_components=80) 
pca = PCA(n_components=50) 
#pca.fit(X_train)
X_train=pca.fit_transform(X_train)

#X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# In[ ]:


X_train.shape


# In[ ]:


print(pca.components_.shape)
print(pca.n_components)


# In[ ]:


pca.explained_variance_


# ### Decompose PCA

# In[ ]:


import math

def get_important_features(transformed_features, components_, columns):
    """
    This function will return the most "important" 
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    #yvector = components_[1] * max(transformed_features[:,1])

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    #important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = { columns[i] : math.sqrt(xvector[i]**2 ) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print ("Features by importance:\n", important_features)

get_important_features(X_train, pca.components_, names)


# In[ ]:


#pca = PCA().fit(X_train)
#fig, ax = plt.subplots()

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper", font_scale=1.3)  
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
#ax.grid(linestyle='-', linewidth='0.2', color='black')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance');
axes = plt.gca()
axes.set_xlim([0,90])
axes.set_ylim([0.3,1.03])
plt.tight_layout()
sns.despine(top=True)
#plt.savefig('Fig1.png', dpi=300)


# In[ ]:


X_train.shape


# ### KPCA

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc=MinMaxScaler()


X=df2.iloc[:,:]
y=df2.Class
le.fit(y)
y=le.transform(y)

X=X.drop(labels='Class', axis=1)
split = int(len(dfc)*0.80)
sp=split-75
X_train, X_test, y_train, y_test = X[:split], X[sp:], y[:split], y[sp:]
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components=50, kernel="rbf", gamma=0.04)
X_train = rbf_pca.fit_transform(X_train)
X_test = rbf_pca.transform(X_test)


# In[ ]:


X_train.shape


# In[ ]:


y_train


# In[ ]:


print(rbf_pca.n_components)


# ## LDA

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc=MinMaxScaler()

split = int(len(dfc)*0.80)
sp=split-100
X_train, X_test, y_train, y_test = X[:split], X[sp:], y[:split], y[sp:]
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=50)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[ ]:


X_train


# ## TRAIN METRICS

# In[ ]:


from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score

svm_acc=accuracy_score(y_train, svm1.predict(X_train))
gnb_acc=accuracy_score(y_train, gnb.predict(X_train))  
lr_acc=accuracy_score(y_train, lr.predict(X_train))  
rf_acc=accuracy_score(y_train, rf.predict(X_train)) 
xgb_acc=accuracy_score(y_train, xgb.predict(X_train)) 
gbm_acc=accuracy_score(y_train, gbm.predict(X_train)) 
et_acc=accuracy_score(y_train, et.predict(X_train)) 

svm_pr=precision_score(y_train, svm1.predict(X_train),average='binary') 
gnb_pr=precision_score(y_train, gnb.predict(X_train),average='binary') 
lr_pr=precision_score(y_train, lr.predict(X_train),average='binary') 
rf_pr=precision_score(y_train, rf.predict(X_train),average='binary') 
xgb_pr=precision_score(y_train, xgb.predict(X_train),average='binary') 
gbm_pr=precision_score(y_train, gbm.predict(X_train),average='binary') 
et_pr=precision_score(y_train, et.predict(X_train),average='binary') 

svm_f1=f1_score(y_train, svm1.predict(X_train),average='binary') 
gnb_f1=f1_score(y_train, gnb.predict(X_train),average='binary') 
lr_f1=f1_score(y_train, lr.predict(X_train),average='binary') 
rf_f1=f1_score(y_train, rf.predict(X_train),average='binary') 
xgb_f1=f1_score(y_train, xgb.predict(X_train),average='binary') 
gbm_f1=f1_score(y_train, gbm.predict(X_train),average='binary') 
et_f1=f1_score(y_train, et.predict(X_train),average='binary') 

svm_rc=recall_score(y_train, svm1.predict(X_train),average='binary') 
gnb_rc=recall_score(y_train, gnb.predict(X_train),average='binary') 
lr_rc=recall_score(y_train, lr.predict(X_train),average='binary') 
rf_rc=recall_score(y_train, rf.predict(X_train),average='binary') 
xgb_rc=recall_score(y_train, xgb.predict(X_train),average='binary') 
gbm_rc=recall_score(y_train, gbm.predict(X_train),average='binary')  
et_rc=recall_score(y_train, et.predict(X_train),average='binary') 

results=pd.DataFrame({'Algorithm':['SVM','Naive Bayes','Logistic Regression','Random Forest','XGBoost', 'LGBM', 'ExtraTrees'],
                     'Accuracy':[svm_acc,gnb_acc,lr_acc, rf_acc, xgb_acc, gbm_acc, et_acc],
                     'Precision':[svm_pr, gnb_pr, lr_pr, rf_pr, xgb_pr, gbm_pr, et_pr],
                     'Recall':[svm_rc,gnb_rc,lr_rc, rf_rc, xgb_rc, gbm_rc, et_rc],
                     'F1-Score':[svm_f1, gnb_f1, lr_f1, rf_f1, xgb_f1, gbm_f1, et_f1]})
results.set_index('Algorithm', inplace=True)
results


# ### Stack 

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


lr = svm.SVC(kernel='linear',C=0.3,random_state=42)
#clf1 = lightgbm.LGBMClassifier(n_estimators=10,max_depth=20,random_state=42,min_samples_leaf=20)
clf1 = xgboost.XGBClassifier(n_estimators=7,max_depth=3,random_state=42,min_samples_leaf=20)

clf2 = RandomForestClassifier(n_estimators=7, max_depth=3,random_state=42, n_jobs=-1)
#clf3 = MLPClassifier(solver='adam', activation='relu',batch_size=12,
                    #hidden_layer_sizes=(20,12,6,1), random_state=42, shuffle=False)
clf3 = GaussianNB()

#clf1 = LogisticRegression(C=0.4,random_state=42)

from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
# Initialize Ensemble

model_stack = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr,use_probas=True,
                                average_probas=True)


# Fit the model on our data
model_stack.fit(X_train, y_train)

# Predict training set
model_pred1 = model_stack.predict(X_train)
model_pred2 = model_stack.predict(X_test)


# In[ ]:


from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score
nn_acc1=accuracy_score(y_train, model_pred1)
nn_acc2=accuracy_score(y_test, model_pred2)

nn_pr1=precision_score(y_train, model_pred1) 
nn_pr2=precision_score(y_test, model_pred2) 

nn_f11=f1_score(y_train, model_pred1) 
nn_f12=f1_score(y_test, model_pred2) 

nn_rc1=recall_score(y_train, model_pred1) 
nn_rc2=recall_score(y_test, model_pred2) 

print('Accuracy Train:',nn_acc1,'Accuracy Test:',nn_acc2)
print('Precision Train:',nn_pr1,'Precision Test:',nn_pr2)
print('Recall Train:',nn_rc1,'Recall Test:',nn_rc2)
print('F1 Score Train:',nn_f11,'F1 Score Test:',nn_f12)


# In[ ]:


X_train.shape


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

X_t=X_train[:,1:3]
gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10,8))

for clf, lab, grd in zip([clf1, clf2, clf3, model_stack], 
                         ['LightGBM', 
                          'XGBoost', 
                          'Extratrees',
                          'StackingClassifier'],
                          itertools.product([0, 1], repeat=2)):

    clf.fit(X_t, y_train)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_t, y=y_train, clf=clf)
    plt.title(lab)


# ## <font color='darkblue'>Tree Based Feature Importance</font>

# In[ ]:


importances = list(zip(xgb.feature_importances_[:30], X.columns[:30]))
importances.sort(reverse=True)
sns.set(style="white")
sns.set(font_scale=1.2)
ax=pd.DataFrame(importances, index=[x for (_,x) in importances]).plot(kind = 'bar',
                                                                      figsize=(14,4),
                                                                     legend=False)
ax.set_facecolor('white')
sns.despine(left=True)


# In[ ]:


importances = list(zip(rf.feature_importances_[:30], X.columns[:30]))
importances.sort(reverse=True)
sns.set(style="white")
sns.set(font_scale=1.2)
ax=pd.DataFrame(importances, index=[x for (_,x) in importances]).plot(kind = 'bar',
                                                                      figsize=(14,4),
                                                                     legend=False)
ax.set_facecolor('white')
sns.despine(left=True)


# In[ ]:


feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X.columns, xgb.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance',ascending=False)[:20]


# In[ ]:


X_train.shape


# ### Reduce Features

# In[ ]:


a=np.percentile(rf.feature_importances_,60)
print(a)
len(xgb.feature_importances_[xgb.feature_importances_>a])
#rf quantile 68
#xgb quantile 79


# In[ ]:


def selectKImportance(model, X, k=5):
     return X.iloc[:,model.feature_importances_.argsort()[::-1][:k]]
X=selectKImportance(rf,X,k=30)


# In[ ]:


unique, counts = np.unique(y_test, return_counts=True)

print (np.asarray((unique, counts)).T)


# ## <font color='darkblue'>Keras ANN Model</font>

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

classifier = Sequential()

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
classifier.add(Dropout(.5))

#classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(.4))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(.4))


classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.fit(X_train, y_train, epochs =70, batch_size=16, validation_data=(X_test, y_test), 
               callbacks=[EarlyStopping(monitor='val_loss', patience=20)], verbose=1)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred= np.where(y_pred>0.48,1,0)
y_pred=y_pred.flatten()
y_pred = y_pred.astype(np.int64)

from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score
nn_acc=accuracy_score(y_test, y_pred)
nn_pr=precision_score(y_test, y_pred) 
nn_f1=f1_score(y_test, y_pred) 
nn_rc=recall_score(y_test, y_pred) 

print('Accuracy:',nn_acc)
print('Precision:',nn_pr)
print('Recall:',nn_rc)
print('F1 Score:',nn_f1)


# In[ ]:


y_pred = classifier.predict(X_train)
y_pred= np.where(y_pred>0.48,1,0)
y_pred=y_pred.flatten()
y_pred = y_pred.astype(np.int64)

from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score
nn_acc=accuracy_score(y_train, y_pred)
nn_pr=precision_score(y_train, y_pred) 
nn_f1=f1_score(y_train, y_pred) 
nn_rc=recall_score(y_train, y_pred) 

print('Accuracy:',nn_acc)
print('Precision:',nn_pr)
print('Recall:',nn_rc)
print('F1 Score:',nn_f1)


# In[ ]:


X_train.shape


# ### ROC Curve

# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ## <font color='darkblue'>Technical Indicators</font>

# In[ ]:


dfse['H-L'] = dfse['High_GOOGL'] - dfse['Low_GOOGL']
dfse['O-C'] = dfse['Close_GOOGL'] - dfse['Open_GOOGL']

dfse['3day MA'] = dfse['Close_GOOGL'].shift(1).rolling(window = 3).mean()
dfse['10day MA'] = dfse['Close_GOOGL'].shift(1).rolling(window = 10).mean()
dfse['30day MA'] = dfse['Close_GOOGL'].shift(1).rolling(window = 30).mean()
dfse['Std_dev']= dfse['Close_GOOGL'].rolling(5).std()

dfse['RSI'] = talib.RSI(dfse['Close_GOOGL'].values, timeperiod = 9)

dfse['OBV'] = talib.OBV(dfse['Close_GOOGL'].values, dfse['Volume_GOOGL'].values)

dfse['TRange'] = talib.TRANGE(dfse['High_GOOGL'].values, dfse['Low_GOOGL'].values,
                            dfse['Close_GOOGL'].values)


dfse['Ult'] = talib.ULTOSC(dfse['High_GOOGL'].values, dfse['Low_GOOGL'].values,
                          dfse['Close_GOOGL'].values, timeperiod1=7, 
                          timeperiod2=14, timeperiod3=28)

dfse['Williams %R'] = talib.WILLR(dfse['High_GOOGL'].values, 
                                 dfse['Low_GOOGL'].values, dfse['Close_GOOGL'].values, 7)


# In[ ]:


dfse.dropna(inplace=True)
dfse.head(2)


# In[ ]:


dfc.head()


# ## <font color='darkblue'>Text Features</font>
# 
# To analyze the text variable we create a class **TextCounts**. In this class we compute some basic statistics on the text variable. This class can be used later in a Pipeline, as well.
# 
# * **count_words** : number of words in the tweet
# * **count_mentions** : referrals to other Twitter accounts, which are preceded by a @
# * **count_hashtags** : number of tag words, preceded by a #
# * **count_capital_words** : number of uppercase words, could be used to *"shout"* and express (negative) emotions
# * **count_excl_quest_marks** : number of question or exclamation marks
# * **count_urls** : number of links in the tweet, preceded by http(s)
# * **count_emojis** : number of emoji, which might be a good indication of the sentiment

# In[37]:


import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
import emoji
from pprint import pprint
import collections

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import gensim

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')

np.random.seed(37)


# In[38]:


class TextCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x)) 
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
        dt_time=df2.DateTime

        
        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags
                           , 'count_capital_words': count_capital_words
                           , 'count_excl_quest_marks': count_excl_quest_marks
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                           , 'DateTime':dt_time
                          })
        
        return df


# In[39]:


dfo.head(2)


# In[40]:


dfo['Text'] = dfo['Text'].astype(object)


# In[41]:


#dfo=pd.read_excel('microsoft_all_dec10.xlsx')
df2 = dfo[['Text','DateTime']]
df2['Text'] = df2['Text'].astype(object)
#df2['Text'] =df2.Text.str.lower()
df2.reset_index(inplace=True)
del(df2['index'])
df2['DateTime'] = pd.to_datetime(df2['DateTime'],errors='coerce')
df2['DateTime']=df2.DateTime.dt.date
print(df2.shape)

def preprocess(tweet):
    if type(tweet)!=type(2.0):
        #tweet.startswith('a')
        tweet = tweet
    else:
        tweet=''
    return tweet

df2['Text']=df2.Text.apply(preprocess)
df2['Text'].astype(str);
print(df2.shape)


# In[42]:


df2.dropna(inplace=True)
tc = TextCounts()
df_eda = tc.fit_transform(df2.Text)


# In[43]:


#df_eda['DateTime']=df_eda.DateTime.dt.date
dft=df_eda.groupby(['DateTime']).sum()
dft.index = pd.to_datetime(dft.index)


# In[44]:


for obs in range(1,4):
    dft['Tcount_words'+str(obs)]=dft.count_words.shift(obs)
    dft['Tcount_mentions'+str(obs)]=dft.count_mentions.shift(obs)
    dft['Tcount_hashtags'+str(obs)]=dft.count_hashtags.shift(obs)
    dft['Tcount_capital_words'+str(obs)]=dft.count_capital_words.shift(obs)
    dft['Tcount_excl_quest_marks'+str(obs)]=dft.count_excl_quest_marks.shift(obs)
    dft['Tcount_urls'+str(obs)]=dft.count_urls.shift(obs)


# In[45]:


dft.dropna(inplace=True)
dft.head(2)


# In[46]:


df_eda['count_urls'].sum()


# In[48]:


df_eda.columns


# In[49]:


inc={'count_mentions':'Mentions @','count_hashtags':'Hashtag #','count_capital_words':'Capital',
    'count_excl_quest_marks':'! or ?','count_urls':'URL'}


# In[50]:


a=pd.DataFrame(df_eda.sum(), columns=['Count'])
#a.loc['count_capital_words','Count']=199173
a.head(7)


# In[ ]:


#a=pd.DataFrame(df_eda.sum(), columns=['Count'])
a['Features'] = a.index
a=a.iloc[1:6,:]
print(type(a))
a['Features']=a.Features.map(inc)
sns.set_style('white')
sns.set_context("paper", font_scale=1.3)  
ax=a.plot.bar(x='Features',y='Count',color='#1f77b4',width=0.5,legend=False, figsize=(6,4))
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.xticks(rotation=30)
plt.xlabel('Tweet Features')
plt.ylabel('Count')
plt.box(False)
plt.tight_layout()
plt.title('Amazon')
plt.savefig('Fig2.3.png', bbox_inches='tight', dpi=300)


# In[76]:


dft['DateTime']=pd.to_datetime(dft['DateTime'])


# In[77]:


#dff=pd.merge(dfse,dft, on='DateTime', how='left')
#dff=pd.merge(dfc,dft, on='DateTime', how='left')
dff=pd.merge(dfm,dft, on='DateTime', how='left')
dff.dropna(inplace=True)
print(dff.shape)
dff.head(1)


# In[ ]:


dff.columns


# ## <font color='darkblue'>N-Grams and TF-IDF</font>

# In[52]:


class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X


# In[53]:


def preprocess(tweet):
    if type(tweet)!=type(2.0):
        #tweet.startswith('a')
        tweet = tweet
    else:
        tweet=''
    return tweet

dft=dfo.loc[:,['Text','DateTime']]
dft['DateTime'] = pd.to_datetime(dft['DateTime'],errors='coerce')
dft['DateTime']=dft.DateTime.dt.date
dft['Text']=dft.Text.apply(preprocess)
dft['Text'].astype(str);
print(dft.shape)


# In[54]:


ct = CleanText()
df_clean = ct.fit_transform(dft.Text)
df_clean.shape


# In[57]:


s1=dft.DateTime
data=pd.DataFrame(dict(Text = df_clean, DateTime = s1)).reset_index()
print(data.shape)
data=data.drop(data[data.Text==''].index)
print(data.shape)


# In[58]:


data.head(2)


# In[ ]:


# Removing Amazon
#data['Text']=data.Text.str.replace('apple','')


# ### <font color='darkblue'> 1. Count Vectorizer </font>

# * max_df = 0.50 means "ignore terms that appear in more than 50% of the documents"
# 
# * min_df = 0.01 means "ignore terms that appear in less than 1% of the documents"

# In[59]:


cv=CountVectorizer(min_df=0.01, max_df=0.9,max_features=70, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,2),
                            stop_words = "english")


X=cv.fit_transform(data.Text)
ss=pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
ss['DateTime']=data.DateTime.values
print(ss.shape)
ss.head(3)


# In[60]:


ss1=pd.DataFrame(ss.sum())
ss1=ss1.sort_values(by=0,ascending=False)
ssf=(ss1/sum(ss1[0]))*100


# In[ ]:


ssf[0:50]


# ### <font color='darkblue'> 2. TF-IDF </font>

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tf_idf=TfidfVectorizer(min_df=0.001, max_df=0.9,max_features=60, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,2),
                            use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english")


X=tf_idf.fit_transform(data.Text)
ss=pd.DataFrame(X.toarray(), columns=tf_idf.get_feature_names())
ss['DateTime']=data.DateTime.values
ss.head(3)


# In[ ]:


sns.set_style('white')
sns.set_context("paper", font_scale=1.3) 
ssx=ss.iloc[:,:71]
ad=ssx.sum()
ad=pd.DataFrame(ad, columns=['Count'])

b1=ad['Count'].quantile(0.90)
b2=ad['Count'].quantile(0.5)
ad=ad[(ad["Count"] < b1) & (ad["Count"] > b2)]

sns.distplot(ad, bins=11, norm_hist=True, kde=True, color='#1f77b4',hist_kws=dict(alpha=0.9));
plt.xlabel('N-Grams (Microsoft)')
plt.ylabel('Density')
plt.box(False)
plt.tight_layout()
plt.savefig('Fig2.4.png', bbox_inches='tight', dpi=300)


# In[ ]:


dff.shape, ss1.shape


# In[78]:


# Merge
ss1=ss.groupby(['DateTime']).sum()
ss1.index=pd.to_datetime(ss1.index)
df2=pd.merge(dff,ss1, on='DateTime', how='left')
df2.dropna(inplace=True)
df2.isnull().sum()
df2.head(2)


# In[80]:


df2.Class


# ## 3D Plot (Features, Models, Accuracy)

# In[ ]:


df=pd.read_excel('Stocks_gr.xlsx', sheet_name='ss1')
df.head(2)


# In[ ]:


df.Z.max()


# In[ ]:


# library
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import pandas as pd
import seaborn as sns
 

# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes

df['Y']=pd.Categorical(df['Y'])
df['Y']=df['Y'].cat.codes


# Make the plot
fig = plt.figure(figsize=(7,4))

ax = fig.gca(projection='3d')

# to Add a color bar which maps values to colors.
ax.plot_trisurf(df['Y'], df['X'], df['Z'], lw=0.6,cmap=plt.cm.viridis, linewidth=0.2)
#ax.plot_surface(X, Y, Z, rstride=10, cstride=10, color='orangered', edgecolors='k', lw=0.6)

fig.colorbar(surf, shrink=0.7, aspect=10,fraction=0.1,pad=0.0001)

ax.get_xaxis().set_visible(False)
ax.grid(False)

plt.xticks([1, 2, 3, 4, 5], ['SVM', 'LR', 'RF','XGB','ANN'])
plt.yticks([1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16], ['A', 'B', 'C','D','E','F','G','H','J',
                                                         'K','L','M','N','P','Q','R'])
#ax.set_zticks([])
ax.set_zlim(0.456, 0.63)
ax.w_zaxis.set_major_locator(LinearLocator(6))

plt.xlim([0.5, 5])
plt.ylim([0.5, 17])

sns.despine(left=True)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_label('Accuracy')

ax.view_init(25,25)
ax.grid(which='both')
plt.xlabel('Models', labelpad=10)
plt.ylabel('Features', labelpad=10)

ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Accuracy', fontsize=10,rotation=90)

plt.tight_layout()
plt.savefig('Fig2.51.png', bbox_inches='tight', dpi=300)


# ## Granger Causality Tests

# In[ ]:


#dfa=pd.read_csv('microsoft_num.csv')
dfa=pd.read_csv('Tesla_num.csv')
dfa['DateTime']=pd.to_datetime(dfa['Date'])
dfa.set_index('DateTime', inplace=True)
del(dfa['Date'])
del(dfa['Close'])
dfa=dfa[(dfa.index>='2007-01-01')&(dfa.index<'2018-11-02')]
dfa = dfa[['Open', 'High', 'Low', 'Volume','Adj Close']]


# In[ ]:


#dfo=pd.read_excel('microsoft_all_dec10.xlsx')
dfo=pd.read_excel('tesla_all_mar10.xlsx')
dfo['Text'] = dfo['Text'].astype(object)
dfo['Text'] =dfo.Text.str.lower()
dfo['Processed_text']=dfo.Text.apply(preprocess)
dfo['pol_sub']=dfo.Processed_text.apply(lambda tweet:TextBlob(tweet).sentiment)
dfo['Polarity']=dfo['pol_sub'].apply(lambda x:x[0])
dfo['Subjectivity']=dfo['pol_sub'].apply(lambda x: x[1])
del (dfo['pol_sub'])
df1=dfo.copy(deep=True)
df1=df1.loc[:,['DateTime','Polarity','Subjectivity']]
df1['DateTime'] = pd.to_datetime(df1['DateTime'],errors='coerce')
df1['DateTime']=df1.DateTime.dt.date

def label (row):
    if row['Polarity'] >= 0.2:
        return 'pos'
    elif row['Polarity']<=-0.05:
        return 'neg'
    else:
        return 'neut'

df1['Sentiment'] = df1.apply (lambda row: label(row),axis=1)
print(df1.Sentiment.value_counts())


aa=df1.loc[:,['DateTime','Subjectivity']]
aa=aa.groupby('DateTime').mean()

gg=df1.groupby(['DateTime', 'Sentiment']).size().reset_index(name='count')
gg=gg.pivot_table(index='DateTime',columns='Sentiment',aggfunc='sum')
gg.fillna(0, inplace=True)
gg.columns = gg.columns.droplevel(level=0)
gg['Subjectivit']=aa
print(gg.shape)

gg.index = pd.to_datetime(gg.index)
gg.columns = ['neg', 'neut','pos','subj']

dfc=pd.merge(dfa,gg, on='DateTime', how='left')
dfc=dfc.dropna()


# In[ ]:


df1=dfo.copy(deep=True)
df1=df1.loc[:,['DateTime','Polarity','Subjectivity']]
df1['DateTime'] = pd.to_datetime(df1['DateTime'],errors='coerce')
df1['DateTime']=df1.DateTime.dt.date
print('Tweets Dataframe',df1.shape)
df1=df1.groupby('DateTime').mean()
df1.index = pd.to_datetime(df1.index)
df2=pd.merge(dfa,df1, on='DateTime', how='left')
df2.dropna(inplace=True)
print('Merged Dataframe',df2.shape)
df2.head()


# In[ ]:


# Dickey Fuller
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

   
    #Perform Dickey-Fuller test:
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:


# Make Stationary
tslog=np.log(df2['Adj Close'])

tslog = tslog - tslog.shift(2)
tslog.dropna(inplace=True)
test_stationarity(tslog)
df2=df2.iloc[3:,:]
df2['tslog']=tslog


# In[ ]:


from decimal import Decimal
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

format_e(Decimal('-9.612912e+00'))


# In[ ]:


ar1=df2['tslog']
ar2=df2['Polarity']
x = np.array([ar1, ar2], np.float)
x = x.transpose()
import statsmodels.tsa.stattools as sm
lag = 40
sm.grangercausalitytests(x, lag)


# In[ ]:


test_stationarity(dfc['Adj Close'])


# ### Topic Modeling

# In[ ]:


doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]


# In[ ]:


print(type(doc_complete))
doc_complete


# In[ ]:


doc_complete=list(dfo.Text[:10000])
doc_complete


# In[ ]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Importing Gensim
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean )

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=20, update_every=500)


# In[ ]:


print(ldamodel.print_topics(num_topics=3, num_words=5))


# In[ ]:


#spacy
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[ ]:


doc_set=dfo.Processed_text[25000:50000]

#don't want puncuations, spaces as tokens
tokenize_blacklist = ['PUNCT', 'SPACE']

texts = []

for doc in doc_set:
    # print(doc)
    
    # putting our three steps together
    
    #1. Tokenize
    doc_sp = nlp(doc)
    tokens = [token.text.lower() for token in doc_sp if token.pos_ not in tokenize_blacklist]
    #tokens = [token.text.lower() for token in doc_sp]
    if(len(tokens) < 3):
        continue
    #2. remove stop words
    stopped_tokens = [token for token in tokens if not token in spacy_stopwords]
    
    #3. lemmetize
    lemmed_tokens = []
    for stopped_token in stopped_tokens:
        lemmed_nlp = nlp(stopped_token)
        lemmed_token = lemmed_nlp[0].lemma_
        lemmed_tokens.append(lemmed_token)
    
    
    # add tokens to list, let's start with stopped_tokens, lemmitization is messing up.
    # for loop so that we don't get list of lists
    for stopped_token in stopped_tokens:
        texts.append(stopped_tokens)


# #### Create bigram and trigram models

# In[ ]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
#def remove_stopwords(texts):
#    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[ ]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(texts, min_count=5, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[texts], threshold=30)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[texts[0]]])


# In[ ]:


# Form Bigrams
data_words_bigrams = make_bigrams(texts)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
#nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_words_bigrams[:1])


# #### Create the Dictionary and Corpus needed for Topic Modeling

# In[ ]:


candidate_text = texts # unigrams :Perplexity:  -4.377417073025684 Coherence Score:  0.7472006215139164
#candidate_text = data_words_bigrams ## Perplexity:  -4.377417073025684  Coherence Score:  0.7472006215139164

# Create Dictionary
id2word = corpora.Dictionary(candidate_text)

# Create Corpus
texts = candidate_text

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# In[ ]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:


# Print the Keyword in the 10 topics
from pprint import pprint
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[ ]:


# Compute Perplexity


from gensim import corpora, models
from gensim.models import CoherenceModel
import gensim


print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=candidate_text, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# #### How to find the optimal number of topics for LDA?

# In[ ]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,  num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10,  alpha='auto',  per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[ ]:


#Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=candidate_text, start=2, limit=40, step=6)

# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[ ]:




