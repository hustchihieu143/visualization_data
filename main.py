# data
import pandas as pd
import numpy as np

# visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# styling
# %matplotlib inline
sns.set_style('darkgrid')
matplotlib.rcParams['figure.facecolor'] = '#00000000'
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# read dataset
df =  pd.read_csv('./most_subscribed_youtube_channels.csv')

# Handle missing values
print(df.shape)
df.isnull().sum()
df.dropna(axis=0,inplace=True)
print(df.shape)

#Convert datatype of the columns
df.columns

#Edit the names of the columns with the last space
df.rename(columns=lambda x: x.strip() , inplace = True)
df.columns
df.dtypes
#Edit data type
df['video views']=df['video views'].str.replace(',','')
df['video count']=df['video count'].str.replace(',','')
df['subscribers']=df['subscribers'].str.replace(',','')
df['video views']=df['video views'].astype('int64')
df['video count']=df['video count'].astype('int64')
df['subscribers']=df['subscribers'].astype('int64')

#Exploratory analysis
print(df.dtypes)

# Data visualization
from IPython.core.display import HTML

#Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
def multi_table(table_list):
        return HTML('<table><tr style="background-color:white;">' +  ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +'</tr></table>')

nunique_df={var:pd.DataFrame(df[var].value_counts())
           for var in {'category', 'started'}}

multi_table([nunique_df['category'],nunique_df['started']])


# Show result

# Categories of Youtube Channel

# categories=df['category'].value_counts()
# fig=px.pie(values=categories.values,
#           names=categories.index,
#           color_discrete_sequence=px.colors.sequential.RdBu,
#           title="Categories of Youtube Channels", template='presentation'
#           )
# fig.update_traces(textposition='inside',
#                   textfont_size=11,
#                   textinfo='percent+label')
# fig.show()


# Started year of youtube channels

# year=df['started'].value_counts()
# plt.figure(figsize=(20,8))
# sns.pointplot(x=year.index,y=year.values, color='violet')
# plt.xlabel('Year')
# plt.ylabel('Count')
# plt.title('Started year of youtube channels',size=30, color='maroon')
# plt.show()



# Mean subscribers, video views and vido count of all youtube channels year by year
# year_mean=df.groupby('started').mean().reset_index()
# year_mean

# def pltplot(data, xcol, ycol, color, ax, title):
#     sns.pointplot(data=data, x=xcol, y=ycol, color=color, ax=ax).set_title(title, size=10)
    

# fig, ((ax1),(ax2),(ax3))=plt.subplots(ncols=1, nrows=3)
# fig.set_size_inches(20,10)
# fig.tight_layout(pad=3.0)

# pltplot(year_mean,'started','subscribers','lightcoral', ax1,'Subscribers per Year (mean)')
# pltplot(year_mean,'started','video views','green', ax2,'Video views per Year (mean)')
# pltplot(year_mean,'started','video count','gold', ax3,'Video count per Year (mean)')



# Top 5 Categories that have subscribers
# subscribers=df.sort_values('subscribers',ascending=False)
# plt.figure(figsize=(25,10))
# subscribers=subscribers[:5]
# sns.barplot(x="category",
#            y="subscribers",
#            data=subscribers,
#            palette="ch:20_r")
# plt.title('Top 5 Categories that have subscribers',size=20)
# plt.show()


#Top 5 Categories that have Video Views
# videoviews=df.sort_values('video views',ascending=False)
# plt.figure(figsize=(25,10))
# videoviews=videoviews[:5]
# sns.barplot(x="category",
#            y="video views",
#            data=videoviews,
#            palette="ch:40_r")
# plt.title('Top 5 Categories that have Video Views',size=20)
# plt.show()


# Top 5 Categories that have Video Counts
# videocount=df.sort_values('video count',ascending=False)
# plt.figure(figsize=(25,10))
# videocount=videocount[:5]
# sns.barplot(x="category",
#            y="video count",
#            data=videocount,
#            palette="ch:30_r")
# plt.title('Top Categories that have Video Counts',size=20)
# plt.show()



#Categories with Video views and Subscribers
# fig = px.scatter(df, x="subscribers", y="video views",
#                  size="video views", color="category",
#                  log_x=True, size_max=50,
#                  title="Categories with Video views and Subscribers",
#                  marginal_y='rug')
# fig.show()



#Categories with Video views and Video count
# fig = px.scatter(df, x="video count", y="video views",
#                  size="video views", color="category",
#                  log_x=True, size_max=50,
#                  title="Categories with Video views and Video count",
#                  marginal_y='rug')
# fig.show()



#Pair plot
# sns.pairplot(df)
# sns.set_theme('notebook')
# plt.show()



#The heat chart depicts the correlation
plt.figure(figsize=(20,8))
sns.set_theme('notebook')
sns.heatmap(df.corr(), annot=True, center=True)
plt.show()