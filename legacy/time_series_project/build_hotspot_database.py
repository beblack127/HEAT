"""Build a provenance-aware SQLite database from available local sources."""
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DB=ROOT/'data'/'hotspot_100.sqlite'

def newest_search_csv():
    files=sorted((ROOT/'data'/'naver').glob('areas_100_*/monthly_search_interest_100areas.csv'))
    return files[-1] if files else None

def main():
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS search_areas (
          search_area TEXT PRIMARY KEY, search_term TEXT NOT NULL,
          sample_group TEXT NOT NULL, admin_dong_code TEXT,
          mapping_status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS search_interest (
          search_area TEXT NOT NULL, month TEXT NOT NULL, search_interest REAL,
          keywords TEXT, source_batch TEXT,
          PRIMARY KEY(search_area,month));
        CREATE TABLE IF NOT EXISTS subway_monthly (
          month TEXT NOT NULL, line TEXT NOT NULL, station TEXT NOT NULL,
          boardings INTEGER NOT NULL, alightings INTEGER NOT NULL,
          observed_days INTEGER NOT NULL,
          PRIMARY KEY(month,line,station));
        CREATE TABLE IF NOT EXISTS area_station_bridge (
          search_area TEXT NOT NULL, line TEXT NOT NULL, station TEXT NOT NULL,
          mapping_rule TEXT NOT NULL, PRIMARY KEY(search_area,line,station));
        CREATE TABLE IF NOT EXISTS living_population_monthly (
          admin_dong_code TEXT NOT NULL, month TEXT NOT NULL,
          value REAL, source_file TEXT, PRIMARY KEY(admin_dong_code,month));
        CREATE TABLE IF NOT EXISTS mobility_monthly (
          admin_dong_code TEXT NOT NULL, month TEXT NOT NULL,
          value REAL, source_file TEXT, PRIMARY KEY(admin_dong_code,month));
        CREATE TABLE IF NOT EXISTS food_business_monthly (
          admin_dong_code TEXT NOT NULL, month TEXT NOT NULL,
          new_licenses INTEGER, closures INTEGER, source_file TEXT,
          PRIMARY KEY(admin_dong_code,month));
        ''')
        area_files=sorted((ROOT/'data'/'naver').glob('areas_100_*/area_list_100.csv'))
        if area_files:
            with area_files[-1].open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            conn.executemany('INSERT OR REPLACE INTO search_areas VALUES (?,?,?,?,?)',
                [(r['search_area'],r['search_term'],r['sample_group'],None,'unverified') for r in rows])
        search_file=newest_search_csv()
        if search_file:
            with search_file.open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            conn.executemany('INSERT OR REPLACE INTO search_interest VALUES (?,?,?,?,?)',
                [(r['search_area'],r['month'],float(r['search_interest']) if r['search_interest'] else None,r['keywords'],r['source_batch']) for r in rows])
        # Existing subway source spans 2024-2025 only. Station/dong mapping requires validation.
        for file in sorted((ROOT/'data'/'subway_raw').glob('CARD_SUBWAY_MONTH_*.csv')):
            encoding='utf-8-sig' if file.read_bytes()[:3]==b'\xef\xbb\xbf' else 'cp949'
            sums=defaultdict(lambda:[0,0,set()])
            with file.open(encoding=encoding,newline='') as f:
                reader=csv.DictReader(f)
                for r in reader:
                    date=r['사용일자']; month=date[:4]+'-'+date[4:6]
                    if month!=file.stem[-6:-2]+'-'+file.stem[-2:]:
                        raise ValueError(f'Month mismatch in {file.name}')
                    line=r.get('노선명') or r.get('호선명')
                    slot=sums[(month,line,r['역명'])]
                    slot[0]+=int(r['승차총승객수']);slot[1]+=int(r['하차총승객수']);slot[2].add(date)
            conn.executemany('INSERT OR REPLACE INTO subway_monthly VALUES (?,?,?,?,?,?)',
                [(month,line,station,board,alight,len(days))
                 for (month,line,station),(board,alight,days) in sums.items()])
        conn.executescript('''
        CREATE VIEW IF NOT EXISTS data_coverage AS
        SELECT 'search_interest' AS source, COUNT(*) AS rows FROM search_interest
        UNION ALL SELECT 'subway_monthly', COUNT(*) FROM subway_monthly
        UNION ALL SELECT 'living_population_monthly', COUNT(*) FROM living_population_monthly
        UNION ALL SELECT 'mobility_monthly', COUNT(*) FROM mobility_monthly
        UNION ALL SELECT 'food_business_monthly', COUNT(*) FROM food_business_monthly;
        ''')
        for source,count in conn.execute('SELECT source,rows FROM data_coverage'):
            print(source,count)
    print(DB)

if __name__=='__main__':main()
