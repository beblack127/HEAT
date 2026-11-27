"""Reproducible pilot panel: trade-area food activity, searches, and spatial audit.

No administrative-dong totals are allocated to named places. Place/trade links
are provisional because place polygons were published after the study period.
"""
from __future__ import annotations
import csv, io, sys, zipfile
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp/pythonlibs'))
import shapefile
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

OUT=ROOT/'results/semester_pilot'
OUT.mkdir(parents=True,exist_ok=True)
REFERENCE=ROOT/'data/reference'

def read_zip_csv(kind,year):
    with zipfile.ZipFile(REFERENCE/f'seoul_commercial_{kind}_{year}.zip') as z:
        data=z.read(z.namelist()[0]).decode('cp949')
    return pd.read_csv(io.StringIO(data),dtype={'상권_코드':str,'trdar_cd':str,'서비스_업종_코드':str,'svc_induty_cd':str},low_memory=False)

def normalize(kind,year):
    df=read_zip_csv(kind,year)
    ren={'기준_년분기_코드':'quarter','stdr_yyqu_cd':'quarter',
         '상권_코드':'trade_code','trdar_cd':'trade_code',
         '상권_코드_명':'trade_name','trdar_cd_nm':'trade_name',
         '서비스_업종_코드':'service_code','svc_induty_cd':'service_code',
         '서비스_업종_코드_명':'service_name','svc_induty_cd_nm':'service_name',
         '점포_수':'store_count','stor_co':'store_count',
         '개업_점포_수':'openings','opbiz_stor_co':'openings',
         '폐업_점포_수':'closures','clsbiz_stor_co':'closures',
         '당월_매출_건수':'sales_transactions','당월_매출_금액':'sales_amount'}
    df=df.rename(columns=ren)
    cols=['quarter','trade_code','trade_name','service_code','service_name']
    cols+=['store_count','openings','closures'] if kind=='stores' else ['sales_transactions','sales_amount']
    df=df[cols]
    df['quarter']=df.quarter.astype(str)
    df['trade_code']=df.trade_code.astype(str)
    return df

def read_footfall():
    frames=[]
    for year in (2023,2024,2025):
        for q in (1,2,3,4):
            x=pd.read_csv(REFERENCE/f'commercial_footfall/footfall_{year}{q}.csv',encoding='cp949',dtype={'상권_코드':str})
            x=x.rename(columns={'기준_년분기_코드':'quarter','상권_코드':'trade_code','총_유동인구_수':'estimated_footfall'})
            x=x[['quarter','trade_code','estimated_footfall']]
            x.quarter=x.quarter.astype(str)
            frames.append(x)
    result=pd.concat(frames,ignore_index=True)
    if result.duplicated(['quarter','trade_code']).any():raise ValueError('Duplicate footfall key')
    return result

def place_trade_bridge():
    path=next((REFERENCE/'seoul_121_place_boundaries').rglob('*.shp'))
    reader=shapefile.Reader(str(path),encoding='utf-8')
    project=Transformer.from_crs(4326,5181,always_xy=True).transform
    place={item.record.as_dict()['AREA_CD']:transform(project,shape(item.shape.__geo_interface__)) for item in reader.iterShapeRecords()}
    with zipfile.ZipFile(REFERENCE/'seoul_commercial_area_2023.zip') as z:
        members={n.rsplit('.',1)[-1].lower():z.read(n) for n in z.namelist() if n.lower().endswith(('.shp','.shx','.dbf'))}
    reader=shapefile.Reader(shp=io.BytesIO(members['shp']),shx=io.BytesIO(members['shx']),dbf=io.BytesIO(members['dbf']),encoding='utf-8')
    trade={str(i.record.as_dict()['TRDAR_CD']):shape(i.shape.__geo_interface__) for i in reader.iterShapeRecords()}
    audit=pd.read_csv(ROOT/'data/place_dong_crosswalk/place_31_mapping_audit.csv',dtype=str).fillna('')
    records=[]
    for row in audit.itertuples():
        if not row.official_place_code or row.training_status=='exclude_shared_polygon': continue
        pg=place[row.official_place_code]
        for code,tg in trade.items():
            if not pg.intersects(tg):continue
            frac=pg.intersection(tg).area/tg.area if tg.area else 0
            if frac>=0.7 and pg.covers(tg.representative_point()):
                records.append({'search_area':row.search_area,'official_place_code':row.official_place_code,
                                'trade_code':code,'trade_area_inside_place':round(frac,5),
                                'name_match_type':row.name_match_type})
    bridge=pd.DataFrame(records)
    if not bridge.empty:
        # A trade must contribute to at most one search place.
        bridge=bridge.sort_values('trade_area_inside_place',ascending=False)
        bridge=bridge.drop_duplicates('trade_code').sort_values(['search_area','trade_code'])
    bridge.to_csv(OUT/'place_trade_bridge_provisional.csv',index=False,encoding='utf-8-sig')
    return bridge

def main():
    stores=pd.concat([normalize('stores',y) for y in (2023,2024,2025)],ignore_index=True)
    sales=pd.concat([normalize('sales',y) for y in (2023,2024,2025)],ignore_index=True)
    for df in (stores,sales):
        df.drop(df[~df.service_code.str.startswith('CS100')].index,inplace=True)
    keys=['quarter','trade_code','trade_name']
    s=stores.groupby(keys,as_index=False)[['store_count','openings','closures']].sum()
    v=sales.groupby(keys,as_index=False)[['sales_transactions','sales_amount']].sum()
    panel=s.merge(v,on=['quarter','trade_code'],how='left',suffixes=('','_sales'),validate='one_to_one')
    panel=panel.drop(columns=['trade_name_sales'])
    panel=panel.merge(read_footfall(),on=['quarter','trade_code'],how='left',validate='one_to_one')
    panel['sales_observed']=panel.sales_transactions.notna().astype(int)
    panel['footfall_observed']=panel.estimated_footfall.notna().astype(int)
    panel.to_csv(OUT/'trade_food_quarterly_2023_2025.csv',index=False,encoding='utf-8-sig')
    bridge=place_trade_bridge()
    search=pd.read_csv(ROOT/'data/naver/areas_20261106T102026Z/monthly_search_interest_31areas.csv')
    search['quarter']=search.month.str[:4]+(((search.month.str[5:7].astype(int)-1)//3)+1).astype(str)
    sq=search.groupby(['search_area','quarter'],as_index=False).agg(search_interest_mean=('search_interest','mean'),search_months=('month','count'))
    sq=sq[sq.search_months==3]
    if len(bridge):
        m=panel.merge(bridge,on='trade_code',how='inner',validate='many_to_one')
        area=m.groupby(['search_area','quarter'],as_index=False).agg(
            trade_count=('trade_code','nunique'),store_count=('store_count','sum'),openings=('openings','sum'),
            closures=('closures','sum'),sales_transactions=('sales_transactions',lambda x:x.sum(min_count=1)),
            sales_amount=('sales_amount',lambda x:x.sum(min_count=1)),
            sales_reporting_trades=('sales_observed','sum'),
            estimated_footfall=('estimated_footfall',lambda x:x.sum(min_count=1)),
            footfall_reporting_trades=('footfall_observed','sum'))
        area=area.merge(sq,on=['search_area','quarter'],how='left',validate='one_to_one')
        area.to_csv(OUT/'named_place_quarterly_provisional.csv',index=False,encoding='utf-8-sig')
    else: area=pd.DataFrame()
    summary={'trade_panel_rows':len(panel),'trade_codes':panel.trade_code.nunique(),
             'quarters':panel.quarter.nunique(),'sales_reporting_rate':round(float(panel.sales_observed.mean()),4),
             'footfall_reporting_rate':round(float(panel.footfall_observed.mean()),4),
             'place_bridge_rows':len(bridge),'named_places_with_trades':bridge.search_area.nunique() if len(bridge) else 0,
             'named_place_panel_rows':len(area)}
    import json
    (OUT/'build_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary)

if __name__=='__main__': main()
