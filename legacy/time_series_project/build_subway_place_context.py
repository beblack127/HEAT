"""Aggregate station exits and link nearby stations to named place polygons.

Station exits are transit context, not counts of visits to the place.
"""
from __future__ import annotations
import csv,re,sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp/pythonlibs'))
import shapefile
from shapely.geometry import Point,shape
from shapely.ops import transform
from pyproj import Transformer
OUT=ROOT/'data/context_monthly';OUT.mkdir(exist_ok=True)

def name(s):return re.sub(r'\([^)]*\)|\s','',s).strip()
def line(s):
    s=s.replace(' ','').replace('9호선2~3단계','9호선(연장)').replace('공항철도1호선','공항철도1호선')
    return s

coords=[]
with (ROOT/'data/reference/station_geom_official.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        try:coords.append({'station_name':name(r['역한글명칭']),'line':line(r['호선명칭']),
                           'lon':float(r['환승역X좌표']),'lat':float(r['환승역Y좌표'])})
        except (ValueError,KeyError):pass
coord=pd.DataFrame(coords)
coord=coord.groupby(['station_name','line'],as_index=False).agg(lon=('lon','mean'),lat=('lat','mean'))
station=defaultdict(lambda:[0,0,0])
files=sorted((ROOT/'data/subway_raw').glob('CARD_SUBWAY_MONTH_202[345][0-1][0-9].csv'))
assert len(files)==36,len(files)
for file in files:
    with file.open('rb') as probe:encoding='utf-8-sig' if probe.read(3)==b'\xef\xbb\xbf' else 'cp949'
    with file.open(encoding=encoding,newline='') as f:
        r=csv.reader(f);next(r)
        for row in r:
            if len(row)<5:continue
            try:off=int(row[4]);on=int(row[3])
            except ValueError:continue
            month=row[0][:4]+'-'+row[0][4:6]
            key=(month,name(row[2]),line(row[1]))
            station[key][0]+=off;station[key][1]+=on;station[key][2]+=1
rows=[{'month':m,'station_name':s,'line':l,'exits':v[0],'entries':v[1],'days':v[2]} for (m,s,l),v in station.items()]
monthly=pd.DataFrame(rows)
monthly=monthly.merge(coord,on=['station_name','line'],how='left',validate='many_to_one')
monthly.to_csv(OUT/'subway_station_monthly_2023_2025.csv',index=False,encoding='utf-8-sig')

project=Transformer.from_crs(4326,5181,always_xy=True).transform
reader=shapefile.Reader(str(next((ROOT/'data/reference/seoul_121_place_boundaries').rglob('*.shp'))),encoding='utf-8')
places={i.record.as_dict()['AREA_CD']:transform(project,shape(i.shape.__geo_interface__)) for i in reader.iterShapeRecords()}
audit=pd.read_csv(ROOT/'data/place_dong_crosswalk/place_31_mapping_audit.csv',dtype=str).fillna('')
audit=audit[(audit.official_place_code!='')&(audit.training_status!='exclude_shared_polygon')]
locations=coord.copy()
locations['point']=locations.apply(lambda r:transform(project,Point(r.lon,r.lat)),axis=1)
bridge=[]
for s in locations.itertuples():
    for a in audit.itertuples():
        dist=places[a.official_place_code].distance(s.point)
        if dist<=500:
            bridge.append({'station_name':s.station_name,'line':s.line,'search_area':a.search_area,
                           'official_place_code':a.official_place_code,'distance_to_place_m':round(dist,1),
                           'association_rule':'each_official_place_within_500m_shared_stations_allowed'})
bridge=pd.DataFrame(bridge)
bridge.to_csv(OUT/'subway_place_bridge_provisional.csv',index=False,encoding='utf-8-sig')
area=monthly.merge(bridge,on=['station_name','line'],how='inner',validate='many_to_many')
area=area.groupby(['search_area','month'],as_index=False).agg(subway_exits=('exits','sum'),subway_entries=('entries','sum'),
                                                    station_lines=('line','count'))
area.to_csv(OUT/'subway_place_monthly_provisional.csv',index=False,encoding='utf-8-sig')
print({'station_lines':len(coord),'monthly_station_rows':len(monthly),'coord_match_rate':round(float(monthly.lon.notna().mean()),3),
       'mapped_station_lines':len(bridge),'mapped_places':bridge.search_area.nunique(),
       'place_month_rows':len(area)})
