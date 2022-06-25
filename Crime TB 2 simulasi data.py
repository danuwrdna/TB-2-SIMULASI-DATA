#!/usr/bin/env python
# coding: utf-8

# In[92]:



import folium


# In[93]:


import pandas as pd


# In[94]:


df = pd.read_csv('crime.csv')


# In[95]:


df.columns = df.columns.str.strip()


# In[96]:


df.head()


# In[97]:


subset_of_df = df.sample(n=1000)


# In[132]:


some_map = folium.Map(location=[subset_of_df['Latitude'].mean(),
                      subset_of_df['Longitude'].mean()],
                               zoom_start=5)


# In[134]:


for row in subset_of_df.itertuples():
    some_map.add_child(folium.Marker(location=[row.Latitude,row.Longitude], popup=[row.TYPE,row.YEAR,row.MONTH,row.DAY,row.HOUR,row.MINUTE ]))


# In[135]:


some_map


# In[106]:


import folium


# In[55]:


some_map_2 = folium.Map(location=[subset_of_df['Latitude'].mean(),
                      subset_of_df['Longitude'].mean()],
                               zoom_start=10)


# In[109]:


icon= folium.Icon()


# In[110]:


for row in subset_of_df.itertuples():
    some_map.add_child(folium.Marker(location=[row.Latitude,row.Longitude], popup=row.TYPE))


# In[112]:


some_map_2


# In[ ]:




