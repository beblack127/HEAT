"""Exploratory search + estimated trade footfall Hotness Score."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/semester_pilot'
df=pd.read_csv(OUT/'named_place_quarterly_provisional.csv',dtype={'quarter':str})
df=df[(df.search_months==3)&(df.footfall_reporting_trades==df.trade_count)&(df.trade_count>=3)].copy()
# Normalize footfall by mapped trade count to reduce arbitrary polygon size effects.
df['footfall_per_trade']=df.estimated_footfall/df.trade_count
df=df.sort_values(['search_area','quarter'])
for col in ('search_interest_mean','footfall_per_trade'):
    df[col+'_yoy']=df.groupby('search_area')[col].pct_change(4,fill_method=None)
    df[col+'_yoy']=df[col+'_yoy'].replace([np.inf,-np.inf],np.nan).clip(-1,3)
base=df[(df.quarter>='20241')&(df.quarter<='20244')]
cols=['search_interest_mean','search_interest_mean_yoy','footfall_per_trade','footfall_per_trade_yoy']
for col in cols:
    ref=base[col].dropna().to_numpy()
    df[col+'_pct']=np.searchsorted(np.sort(ref),df[col].to_numpy(),side='right')/len(ref)*100
df['score_valid']=df[cols].notna().all(axis=1)
df['hotness_score']=0.35*df.search_interest_mean_pct+0.15*df.search_interest_mean_yoy_pct+0.35*df.footfall_per_trade_pct+0.15*df.footfall_per_trade_yoy_pct
df['component_floor_ok']=(df.search_interest_mean_pct>=50)&(df.footfall_per_trade_pct>=50)
df['above_80']=df.score_valid&(df.hotness_score>=80)&df.component_floor_ok
df['prev_above_80']=df.groupby('search_area').above_80.shift(1).fillna(False)
df['persistent_hotspot']=df.above_80&df.prev_above_80
df['prior_persistent_hotspot']=df.groupby('search_area').persistent_hotspot.shift(1).fillna(False)
df['newly_emerged']=df.persistent_hotspot&~df.prior_persistent_hotspot
df['score_search_40']=0.28*df.search_interest_mean_pct+0.12*df.search_interest_mean_yoy_pct+0.42*df.footfall_per_trade_pct+0.18*df.footfall_per_trade_yoy_pct
df['score_search_60']=0.42*df.search_interest_mean_pct+0.18*df.search_interest_mean_yoy_pct+0.28*df.footfall_per_trade_pct+0.12*df.footfall_per_trade_yoy_pct
df['threshold_75']=df.hotness_score>=75
df['threshold_85']=df.hotness_score>=85
df.loc[~df.score_valid,['hotness_score','score_search_40','score_search_60']]=np.nan
df.to_csv(OUT/'named_place_hotness_score_quarterly.csv',index=False,encoding='utf-8-sig')
latest=df[(df.quarter=='20254')&df.score_valid].sort_values('hotness_score',ascending=False)
latest[['search_area','trade_count','search_interest_mean','footfall_per_trade','hotness_score','component_floor_ok','persistent_hotspot','score_search_40','score_search_60']].to_csv(OUT/'hotness_score_2025Q4_rank.csv',index=False,encoding='utf-8-sig')
out={'eligible_places_2025Q4':len(latest),'min_mapped_trade_areas':3,
     'persistent_hotspots_2025Q4':latest[latest.persistent_hotspot].search_area.tolist(),
     'top_5':latest[['search_area','hotness_score']].head(5).round(2).to_dict('records'),
     'warning':'Footfall is an official estimate, not observed visits. Spatial boundaries provisional; score is a pilot and not externally validated.'}
(OUT/'hotness_score_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
