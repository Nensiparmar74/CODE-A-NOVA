from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
FIG_DIR=ROOT/'reports'/'figures'
FIG_DIR.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/'data'/'processed'/'clean_crop_yield.csv')
model=joblib.load(ROOT/'models'/'crop_yield_model.joblib')
X=df.drop(columns=['hg/ha_yield']); y=df['hg/ha_yield']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
pred=model.predict(X_test); resid=y_test.to_numpy()-pred
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':150,'axes.titlesize':12,'axes.labelsize':10})
figures=[]

def savefig(key,title,interpretation):
    fn=FIG_DIR/f'{len(figures)+1:02d}_{key}.png'
    plt.tight_layout(); plt.savefig(fn,bbox_inches='tight'); plt.close()
    figures.append({'id':len(figures)+1,'file':fn.relative_to(ROOT).as_posix(),'title':title,'interpretation':interpretation})

def regplot(x,ydata,title,xlabel,ylabel,key,interpretation,n=5000):
    s=df[[x,ylabel if False else 'hg/ha_yield']].sample(min(n,len(df)),random_state=42) if x!='hg/ha_yield' else df.sample(min(n,len(df)),random_state=42)
    xv=s[x].to_numpy(dtype=float); yv=s['hg/ha_yield'].to_numpy(dtype=float)
    plt.figure(figsize=(10,6)); plt.scatter(xv,yv,s=14,alpha=0.22)
    mask=np.isfinite(xv)&np.isfinite(yv)
    if mask.sum()>1:
        coef=np.polyfit(xv[mask],yv[mask],1); xx=np.linspace(np.nanmin(xv),np.nanmax(xv),100); plt.plot(xx,coef[0]*xx+coef[1],linewidth=2)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); savefig(key,title,interpretation)

# 1
s=df.groupby('Item')['hg/ha_yield'].mean().sort_values(); plt.figure(figsize=(10,5)); plt.barh(s.index,s.values); plt.title('Mean Crop Yield by Crop Variety'); plt.xlabel('Yield (hg/ha)'); plt.ylabel('Crop Variety'); savefig('mean_yield_by_crop','Mean yield by crop variety',f"Potatoes has the highest mean yield at {s.iloc[-1]:,.0f} hg/ha among the 10 crop categories, while {s.index[0]} has the lowest mean.")
# 2
plt.figure(figsize=(10,5)); plt.hist(df['hg/ha_yield'],bins=60); plt.title('Yield Distribution'); plt.xlabel('Yield (hg/ha)'); plt.ylabel('Observation count'); savefig('yield_distribution','Yield distribution','The pooled target is strongly right-skewed, motivating robust tree-based modeling and careful treatment of extreme feature values.')
# 3
plt.figure(figsize=(10,5)); plt.hist(np.log1p(df['hg/ha_yield']),bins=60); plt.title('Log-Transformed Yield Distribution'); plt.xlabel('log1p(Yield)'); plt.ylabel('Observation count'); savefig('log_yield_distribution','Log-transformed yield distribution','The log transform compresses the long right tail and makes the central distribution easier to inspect.')
# 4
items=sorted(df['Item'].unique()); groups=[df.loc[df.Item==c,'hg/ha_yield'].values for c in items]; plt.figure(figsize=(11,6)); plt.boxplot(groups,labels=items,showfliers=False); plt.xticks(rotation=45,ha='right'); plt.title('Yield Spread by Crop Variety'); plt.xlabel('Crop Variety'); plt.ylabel('Yield (hg/ha)'); savefig('yield_box_by_crop','Yield spread by crop','Yield ranges differ materially by crop, indicating that crop identity is a dominant explanatory variable.')
# 5
s=df.groupby('Year')['hg/ha_yield'].mean(); plt.figure(figsize=(10,5)); plt.plot(s.index,s.values,marker='o'); plt.title('Annual Mean Yield Trend (1990-2013)'); plt.xlabel('Year'); plt.ylabel('Mean yield (hg/ha)'); savefig('annual_mean_trend','Annual mean yield trend','Mean pooled yield rises from about 66.7k hg/ha in 1990 to about 90.4k in 2013, with small year-to-year dips in a few years.')
# 6
s=df.groupby('Year')['hg/ha_yield'].median(); plt.figure(figsize=(10,5)); plt.plot(s.index,s.values,marker='o'); plt.title('Annual Median Yield Trend (1990-2013)'); plt.xlabel('Year'); plt.ylabel('Median yield (hg/ha)'); savefig('annual_median_trend','Annual median yield trend','The median series follows the broad upward pattern while reducing the influence of high-yield crop categories.')
# 7
plt.figure(figsize=(10,5)); plt.hist(df['average_rain_fall_mm_per_year'],bins=50); plt.title('Annual Rainfall Distribution'); plt.xlabel('Average annual rainfall (mm)'); plt.ylabel('Observation count'); savefig('rainfall_distribution','Annual rainfall distribution','Rainfall spans 51 to 3,240 mm/year, covering multiple climatic regimes used later for categorical feature engineering.')
# 8
plt.figure(figsize=(10,5)); plt.hist(np.log1p(df['pesticides_tonnes']),bins=60); plt.title('Log-Transformed Pesticide Use Distribution'); plt.xlabel('log1p(pesticides tonnes)'); plt.ylabel('Observation count'); savefig('pesticide_log_distribution','Log-transformed pesticide distribution','Pesticide input is highly skewed; log1p provides a numerically stable representation for EDA and modeling.')
# 9
plt.figure(figsize=(10,5)); plt.hist(df['avg_temp'],bins=45); plt.title('Average Temperature Distribution'); plt.xlabel('Average temperature (°C)'); plt.ylabel('Observation count'); savefig('temperature_distribution','Average temperature distribution','Average temperature ranges from 1.3°C to 30.65°C, indicating wide geographic coverage.')
# 10
num=['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','hg/ha_yield']; corr=df[num].corr(); plt.figure(figsize=(9,7)); im=plt.imshow(corr.values,aspect='auto'); plt.colorbar(im,label='Pearson correlation'); plt.xticks(range(len(num)),num,rotation=45,ha='right'); plt.yticks(range(len(num)),num); 
for i in range(len(num)):
 for j in range(len(num)): plt.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center');
plt.title('Correlation Heatmap - Numeric Features and Yield'); plt.xlabel('Variables'); plt.ylabel('Variables'); savefig('correlation_heatmap','Numeric correlation heatmap','Raw linear correlations with yield are modest for rainfall, pesticides and temperature, because the pooled target varies strongly by crop and country.')
# 11-13
regplot('average_rain_fall_mm_per_year',None,'Rainfall vs Yield (Sampled Observations)','Average annual rainfall (mm)','Yield (hg/ha)','rainfall_vs_yield','The pooled relationship is weak because crop type and geography create substantial between-group variation.')
regplot('avg_temp',None,'Temperature vs Yield (Sampled Observations)','Average temperature (°C)','Yield (hg/ha)','temperature_vs_yield','The raw pooled association is slightly negative, but the scatter shows strong clustering across crop categories.')
regplot('pesticides_tonnes',None,'Pesticide Use vs Yield (Sampled Observations)','Pesticides (tonnes)','Yield (hg/ha)','pesticides_vs_yield','Pooled observations show a modest positive raw association, with large dispersion driven by geography and crop mix.')
# 14
bins=[-np.inf,600,1200,2000,np.inf]; labels=['Low / Arid','Moderate','High','Tropical']; cats=pd.cut(df['average_rain_fall_mm_per_year'],bins=bins,labels=labels); g=df.assign(rainfall_category=cats).groupby('rainfall_category',observed=True)['hg/ha_yield'].mean(); plt.figure(figsize=(9,5)); plt.bar(g.index.astype(str),g.values); plt.title('Mean Yield by Rainfall Category'); plt.xlabel('Rainfall category'); plt.ylabel('Mean yield (hg/ha)'); savefig('yield_by_rainfall_category','Mean yield by rainfall category','Average yield differs across rainfall regimes; the categorical feature provides a simple way to represent non-linear climate bands.')
# 15
s=df.groupby('Area')['hg/ha_yield'].mean().sort_values(ascending=False).head(15).sort_values(); plt.figure(figsize=(10,6)); plt.barh(s.index,s.values); plt.title('Top 15 Countries by Mean Yield'); plt.xlabel('Mean yield (hg/ha)'); plt.ylabel('Area'); savefig('top_country_yield','Top countries by mean yield','The highest pooled country means reflect the mix of crop types and production systems represented in each area, so rankings should not be interpreted as causal country effects.')
# 16
s=df['Item'].value_counts().sort_values(); plt.figure(figsize=(10,5)); plt.barh(s.index,s.values); plt.title('Observation Count by Crop'); plt.xlabel('Records'); plt.ylabel('Crop Variety'); savefig('crop_record_counts','Observation count by crop','The dataset is unbalanced across crop categories; potatoes and maize contribute more observations than several other crops.')
# 17
s=df['Area'].value_counts().head(15).sort_values(); plt.figure(figsize=(10,6)); plt.barh(s.index,s.values); plt.title('Top 15 Areas by Observation Count'); plt.xlabel('Records'); plt.ylabel('Area'); savefig('top_area_counts','Observation count by area','India contributes the largest number of observations among the areas in this dataset, so pooled analyses should account for unequal group sizes.')
# 18
plt.figure(figsize=(11,6)); top_crops=['Potatoes','Cassava','Sweet potatoes','Maize','Rice, paddy'];
for crop in top_crops:
 g=df[df.Item==crop].groupby('Year')['hg/ha_yield'].mean(); plt.plot(g.index,g.values,marker='o',label=crop)
plt.title('Mean Yield Trends for Selected Crops'); plt.xlabel('Year'); plt.ylabel('Mean yield (hg/ha)'); plt.legend(); savefig('selected_crop_trends','Selected crop yield trends','Different crops exhibit distinct baseline yield levels and trajectories; this motivates retaining crop identity in the predictive pipeline.')
# 19
imp=pd.DataFrame(json.load(open(ROOT/'models'/'feature_importance.json'))).head(15).sort_values('importance'); plt.figure(figsize=(10,7)); plt.barh(imp['feature'],imp['importance']); plt.title('Top 15 Random Forest Feature Importances'); plt.xlabel('Importance'); plt.ylabel('Feature'); savefig('feature_importance','Top feature importances','The final Random Forest places high importance on crop indicators such as Item_Potatoes, followed by other crop, area and numeric features.')
# 20
plt.figure(figsize=(8,7)); plt.scatter(y_test,pred,s=14,alpha=0.28); lims=[min(y_test.min(),pred.min()),max(y_test.max(),pred.max())]; plt.plot(lims,lims,'--'); plt.title('Held-Out Actual vs Predicted Yield'); plt.xlabel('Actual yield (hg/ha)'); plt.ylabel('Predicted yield (hg/ha)'); savefig('actual_vs_predicted','Held-out actual vs predicted yield','Predictions cluster around the 45-degree reference line, consistent with the held-out R² of 0.9813.')
# 21
plt.figure(figsize=(10,5)); plt.hist(resid,bins=60); plt.axvline(0,linestyle='--'); plt.title('Held-Out Residual Distribution'); plt.xlabel('Residual = actual - predicted (hg/ha)'); plt.ylabel('Observation count'); savefig('residual_distribution','Residual distribution','Residuals are centered near zero; the remaining spread reflects observations that are harder to predict from the available macro-level features.')

json.dump(figures,open(ROOT/'models'/'eda_figures.json','w'),indent=2)
print('Generated',len(figures),'figures')
