#!/usr/bin/env python
# coding: utf-8

# # Question 1
# 1.Import required libraries and read the provided dataset (youtube_dislike_dataset.csv) and retrieve top5 and bottom 5 records.

# In[1]:


import pandas as pd
youtube_dislike = pd.read_csv("youtube_dislike_dataset.csv")
df = pd.DataFrame(youtube_dislike)
df


# In[2]:


df.head()


# In[3]:


df.tail()


# # Question 2
# 2.Check the info of the dataframe and write your inferences on data types and shape of the dataset.

# In[3]:


df.info()
df.shape


# # My Inferences on the above datatypes and shape of the dataset.
# 
# In the above youtube dataset, we have totally 37422 entries with overall 12 data columns (Video_id, title, channel_id, channel_title, published_at, view_count, likes, dislikes, comment_count, tags, description, comments). Among 12 variables 
# integer variables are 4 and object variables are 8. It has the memory usage of 3.4+ mb. 
# 
# The shape of the dataset is having 37422 rows and 12 columns.

# # Question 3
# 3.Check for the Percentage of the missing values and drop or impute them

# In[5]:


missing_value = df.isnull().sum()
missing_value


# In[6]:


percentage_of_missing_values = (missing_value/len(df)*100)
percentage_of_missing_values


# In[7]:


# imputing the missing values
# impute numerical columns with mean
num_cols = df.select_dtypes(exclude = "O").columns
df[num_cols]=df[num_cols].fillna(df[num_cols].mean())

# impute categorical columns with mode
cat_cols = df.select_dtypes(include = "O").columns
df[cat_cols]=df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Check if any missing values are left
print(df.isnull().sum())


# In[8]:


missing_value = df.isnull().sum()
missing_value


# # Question 4
# 4.Check the statistical summary of both numerical and categorical columns and write your inferences.

# In[10]:


# For both numeric and categorical 

df.describe(include = 'all')


# # Question 5
# 5.Convert datatype of column published_at from object to pandas datetime.

# In[11]:


pd.to_datetime(df['published_at'])


# # Question 6
# 6.Create a new column as 'published_month' using the column published_at (display the months only)

# In[12]:


df['published_month'] = df['published_at'].str[5:7]
df['published_month']


# # Question 7
# 7.Replace the numbers in the column published_month as names of the months i,e., 1 as 'Jan', 2 as 'Feb'
# and so on.....

# In[13]:


month = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug',
         '09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
month


# In[14]:


df['published_month']= df['published_month'].replace(month)
df['published_month']


# # Question 8
# 8.Find the number of videos published each month and arrange the months in a decreasing order based
# on the video count.

# In[15]:


pd.DataFrame(df.groupby('published_month')['video_id'].count().sort_values(ascending=False))


# # Question 9
# 9.Find the count of unique video_id, channel_id and channel_title.

# In[15]:


len(df['video_id'].unique()),len(df['channel_id'].unique()),len(df['channel_title'].unique())


# # Question 10
# 10.Find the top10 channel names having the highest number of videos in the dataset and the bottom10
# having lowest number of videos.

# In[17]:


pd.DataFrame(df.groupby('channel_title')['video_id'].count().sort_values(ascending = False).head(10))


# In[18]:


pd.DataFrame(df.groupby('channel_title')['video_id'].count().sort_values(ascending = False).tail(10))


# # Question 11
# 11.Find the title of the video which has the maximum number of likes and the title of the video having
# minimum likes and write your inferences

# In[19]:


pd.DataFrame(df.groupby('title')['likes'].max().sort_values(ascending=False).head(1))


# In[20]:


pd.DataFrame(df.groupby('title')['likes'].min().sort_values(ascending=False).tail(1))


# # My Inferences for 11th Question
# * The video title (BTS()'Dynamite' Official MV) has the maximum number of likes
# * The video title (Kim Kardashian's Must-See Moments on "Saturday Night Live"|E! News) has the minimum number of likes.

# # Question 12
# 12.Find the title of the video which has the maximum number of dislikes and the title of the video having
# minimum dislikes and write your inferences.

# In[21]:


pd.DataFrame(df.groupby('title')['dislikes'].max().sort_values(ascending=False).head(1))


# In[22]:


pd.DataFrame(df.groupby('title')['dislikes'].min().sort_values(ascending=False).tail(1))


# # My Inferences for 12th Question
# * The video title (Cuties|Official Trailer| Netflix) has the maximum number of dislikes
# * The video title (Kim Kardashian's Must-See Moments on "Saturday Night Live"|E! News) has the minimum number of dislikes. 

# # Question 13
# 13.Does the number of views have any effect on how many people disliked the video? Support your
# answer with a metric and a plot.

# In[24]:


corr = df.corr(numeric_only=True)
corr


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='view_count', y='dislikes', alpha=0.7)
plt.title('View Count vs. Dislikes')
plt.xlabel('View Count')
plt.ylabel('Dislikes')
plt.grid(True)


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('correlaton heatmap')
plt.xlabel('View Count')
plt.ylabel('Dislikes')
plt.show()


# * The correlation stands between -1 and +1
# * The correlation between the view_count and dilikes have strong positive linear relationship (0.68). 
# * Hence the correlation suggests that as the number of views for a video increases,
# the number of dislikes tends to increase as well

# # Question 14
# 14.Display all the information about the videos that were published in January, and mention the count of
# videos that were published in January.

# In[33]:


df[df['published_month']=='Jan']


# In[47]:


df[df['published_month']=='Jan']['video_id'].count()


# In[ ]:





# In[ ]:




