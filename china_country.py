#! /usr/bin/env python

# coding: utf-8

# In[6]:
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np


#data = {'jing':[],'wei':[]}
#with open('dataset/usedata/high_16010108.000','r') as f0:
#    for l in f0:
#        ll = l.split()
#        if 70 < float(ll[1]) < 140 and 15 < float(ll[2]) < 55 and float(ll[4]) <= 8:
#            data['jing'].append(ll[1])
#            data['wei'].append(ll[2])
#df = pd.DataFrame(data)
#df.to_csv('dataset/usedata/surface_point2.csv',sep='\t',index=False)



#high.png
posi=pd.read_csv('high_point.csv',sep='\t')
lat = np.array(posi["wei"])                        
lon = np.array(posi["jing"])  
map = Basemap(lat_0=35, lon_0=110,
              llcrnrlon=70, 
              llcrnrlat=3.01, 
              urcrnrlon=138.16, 
              urcrnrlat=60,resolution='l',area_thresh=1000000,rsphere=6371200.)
map.drawcoastlines()   
map.drawcountries()    
map.drawmapboundary()
parallels = np.arange(5,55,5) 
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=5) # 绘制纬线

meridians = np.arange(70,140.,10.)
map.drawmeridians(meridians,fontsize=5,linewidth=0) # 绘制经线

x,y = map(lon,lat)

map.scatter(x,y,edgecolors='r',facecolors='r',marker='*',s=10)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

plt.title("Distribution of high altitude stations")

plt.xticks([70,80,90,100,110,120,130,140],
           ['$70E$','$80E$','$90E$','$100E$','$110E$','$120E$','$130E$','$140E$'])  # 设置x刻度
plt.yticks([15,20,25,30,35,40,45,50,55],
           ['$15N$','$20N$','$25N$','$30N$','$35N$','$40N$','$45N$','$50N$','$55N$']) # 设置y刻度

plt.xlabel("Longitude",font1)
plt.ylabel("Latitude",font2)
plt.savefig("surface1.png")
plt.show()


posi2=pd.read_csv('surface_point2.csv',sep='\t')
lat2 = np.array(posi2["wei"])                        
lon2 = np.array(posi2["jing"]) 

map = Basemap(lat_0=35, lon_0=110,
              llcrnrlon=70, 
              llcrnrlat=3.01, 
              urcrnrlon=138.16, 
              urcrnrlat=60,resolution='l',area_thresh=1000000,rsphere=6371200.)
map.drawcoastlines()   
map.drawcountries()    
map.drawmapboundary()
parallels = np.arange(5,55,5) 
map.drawparallels(parallels,fontsize=5,linewidth=0) # 绘制纬线

meridians = np.arange(70,140.,10.)
map.drawmeridians(meridians,fontsize=5,linewidth=0) # 绘制经线

x2,y2 = map(lon2,lat2)

map.scatter(x2,y2,edgecolors='blue',facecolors='blue',marker='*',s=10)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

plt.title("Distribution of surface stations")
plt.xticks([70,80,90,100,110,120,130,140],
           ['$70E$','$80E$','$90E$','$100E$','$110E$','$120E$','$130E$','$140E$'])  # 设置x刻度
plt.yticks([15,20,25,30,35,40,45,50,55],
           ['$15N$','$20N$','$25N$','$30N$','$35N$','$40N$','$45N$','$50N$','$55N$']) # 设置y刻度

plt.xlabel("Longitude",font1)
plt.ylabel("Latitude",font2)

plt.savefig("surface2.png")
plt.show()
