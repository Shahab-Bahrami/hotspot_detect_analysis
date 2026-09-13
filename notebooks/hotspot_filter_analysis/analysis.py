"""Auditable matching and rule-testing helpers for the companion notebooks.

All spatial distances use great-circle distances on a sphere (radius 6371.0088 km).
Satellite acquisition times are UTC; customer times are retained as recorded.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_KM = 6371.0088
YEARS = [2019, 2023]
COLS = ['Longitude', 'Latitude', 'Tgl_Hotspot', 'Sumber', 'Satellite', 'Status', 'Region']
SAT_MAP = {'N':'SNPP', 'N20':'NOAA20', 'Aqua':'Aqua', 'Terra':'Terra'}

def xyz(latitude, longitude):
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def chord(km):
    return 2*np.sin(km/(2*EARTH_KM))

def arc(ch):
    return 2*EARTH_KM*np.arcsin(np.clip(ch/2, 0, 1))

def load_inputs(root):
    customer_parts=[]
    for y in YEARS:
        d=pd.read_excel(root/'data/customer/hotspot_master_2019_2023_2025_2026.xlsx', sheet_name=str(y))
        d['excel_row']=np.arange(len(d))+2
        d['sheet']=str(y)
        d=d.dropna(subset=COLS, how='all').copy()
        d['customer_id']=[f'{y}:{r}' for r in d.excel_row]
        d['year']=pd.to_datetime(d.Tgl_Hotspot).dt.year
        assert d.year.eq(y).all()
        d['Satellite']=d.Satellite.fillna('Unknown').astype(str).str.strip()
        customer_parts.append(d)
    customer=pd.concat(customer_parts, ignore_index=True)
    assert not customer.duplicated(COLS).any(), 'Resolve exact customer duplicates before analysis.'
    parts=[]; inventory=[]
    for p in sorted((root/'data/satelites').glob('*/*/*.csv')):
        y=int(p.parent.name)
        if y not in YEARS: continue
        d=pd.read_csv(p, dtype={'acq_time':str, 'confidence':str, 'version':str})
        original=len(d)
        d=d.drop_duplicates().copy()
        d['satellite_name']=d.satellite.map(SAT_MAP)
        assert d.satellite_name.notna().all()
        d['utc']=pd.to_datetime(d.acq_date+' '+d.acq_time.str.zfill(4), format='%Y-%m-%d %H%M')
        assert d.utc.dt.year.eq(y).all()
        d['year']=y
        d['sat_id']=[f'{p.parent.parent.name}/{y}:{i+2}' for i in d.index]
        d['source_file']=str(p.relative_to(root))
        d['confidence_score']=pd.to_numeric(d.confidence,errors='coerce')
        d.loc[d.instrument.eq('VIIRS'),'confidence_score']=d.loc[d.instrument.eq('VIIRS'),'confidence'].map({'l':0,'n':50,'h':100})
        # Ordinal VIIRS scores are a convenience, not calibrated probabilities.
        d['brightness_main']=d.get('bright_ti4',d.get('brightness'))
        d['brightness_background']=d.get('bright_ti5',d.get('bright_t31'))
        inventory.append(dict(file=str(p.relative_to(root)),year=y,rows=original,duplicates_removed=original-len(d),
                              satellites=','.join(sorted(d.satellite_name.unique())),
                              first_utc=str(d.utc.min()),last_utc=str(d.utc.max()),
                              versions=','.join(sorted(d.version.unique())),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        parts.append(d)
    sat=pd.concat(parts,ignore_index=True)
    sat['sid']=np.arange(len(sat))
    assert sat[['latitude','longitude','utc']].notna().all().all()
    return customer,sat,pd.DataFrame(inventory)

def customer_bbox(customer, pad_degrees=0):
    return (customer.Longitude.min()-pad_degrees, customer.Latitude.min()-pad_degrees,
            customer.Longitude.max()+pad_degrees, customer.Latitude.max()+pad_degrees)

def within_bbox(sat,bounds):
    west,south,east,north=bounds
    return sat.longitude.between(west,east)&sat.latitude.between(south,north)

def match_candidates(customer,sat,radius_km=2):
    """All satellite candidates within radius and adjacent UTC calendar dates.

    Do not select on customer fire verification status. Keep other-satellite
    candidates separately for explicit source-label diagnostics.
    """
    chunks=[]
    for y in YEARS:
        c=customer.loc[customer.year.eq(y)]
        s=sat.loc[sat.year.eq(y)&within_bbox(sat,customer_bbox(c,0.05))].copy()
        for date,cg in c.groupby(c.Tgl_Hotspot.dt.normalize(),sort=True):
            sg=s.loc[s.utc.ge(date-pd.Timedelta(days=1))&s.utc.lt(date+pd.Timedelta(days=2))]
            if sg.empty: continue
            points=xyz(sg.latitude.to_numpy(),sg.longitude.to_numpy())
            queries=xyz(cg.Latitude.to_numpy(),cg.Longitude.to_numpy())
            neighbors=cKDTree(points).query_ball_point(queries,chord(radius_km))
            sizes=np.array([len(a) for a in neighbors])
            if not sizes.sum(): continue
            ci=np.repeat(np.arange(len(cg)),sizes)
            si=np.concatenate(neighbors).astype(int)
            cd=cg.iloc[ci]; sd=sg.iloc[si]
            chunks.append(pd.DataFrame({
                'customer_index':cd.index.to_numpy(), 'sid':sd.sid.to_numpy(),
                'distance_km':arc(np.linalg.norm(queries[ci]-points[si],axis=1)),
                'delta_minutes':(cd.Tgl_Hotspot.to_numpy()-sd.utc.to_numpy())/np.timedelta64(1,'m'),
                'same_satellite':cd.Satellite.to_numpy()==sd.satellite_name.to_numpy(),
                'unknown_satellite':cd.Satellite.eq('Unknown').to_numpy(),
            }))
    return pd.concat(chunks,ignore_index=True)

def time_mask(pairs,offset=7,mode='floor'):
    # customer timestamp minus (UTC + offset); customer timestamp is hour-only.
    residual=pairs.delta_minutes.to_numpy()-60*offset
    if mode=='floor': return (residual<=0)&(residual>-60)
    if mode=='nearest': return np.abs(residual)<=30
    if mode=='plusminus60': return np.abs(residual)<=60
    raise ValueError(mode)

def select_matches(pairs,radius_km=0.1,offset=7,mode='floor',allow_unknown=False,any_satellite=False):
    source_ok=pairs.same_satellite.to_numpy()
    if allow_unknown: source_ok=source_ok|pairs.unknown_satellite.to_numpy()
    if any_satellite: source_ok=np.ones(len(pairs),bool)
    p=pairs.loc[source_ok & time_mask(pairs,offset,mode) & pairs.distance_km.le(radius_km)].copy()
    p['time_residual_minutes']=p.delta_minutes-60*offset
    p['abs_time_residual']=p.time_residual_minutes.abs()
    p=p.sort_values(['customer_index','distance_km','abs_time_residual','sid'])
    p['candidate_count']=p.groupby('customer_index').sid.transform('size')
    return p.drop_duplicates('customer_index').drop(columns='abs_time_residual')

def match_audit(customer,pairs):
    rows=[]
    for mode in ['floor','nearest','plusminus60']:
        for offset in range(-12,15):
            m=select_matches(pairs,0.1,offset,mode)
            for y in YEARS:
                n=int(customer.year.eq(y).sum())
                k=int(customer.loc[m.customer_index,'year'].eq(y).sum())
                rows.append(dict(year=y,mode=mode,utc_offset_hours=offset,matched=k,customer_rows=n,coverage=k/n))
    return pd.DataFrame(rows)

def neighbor_features(sat,radii=(0.5,1.0,2.0),windows=(0,10,30,60,1440),progress=False):
    """Count other detections inside exact spatial/time thresholds.

    Each unordered pair is processed once, including pairs across midnight.
    Equal acquisition minutes count as simultaneous detections, not a later visit.
    Compute before restricting scoring to the unbuffered region.
    """
    s=sat.reset_index(drop=True)
    n=len(s); points=xyz(s.latitude.to_numpy(),s.longitude.to_numpy())
    minutes=s.utc.to_numpy().astype('datetime64[m]').astype(np.int64)
    day=minutes//1440; names=s.satellite_name.to_numpy()
    counts={f'n_{r:g}km_{w}min':np.zeros(n,np.int32) for r in radii for w in windows}
    for key in ['n_same_sat_1km_30min','n_other_sat_1km_30min','n_prior_strict_1km_30min','n_different_time_1km_30min']:
        counts[key]=np.zeros(n,np.int32)
    groups={k:np.flatnonzero(day==k) for k in np.unique(day)}
    for iteration,(d,current) in enumerate(groups.items()):
        ids=np.concatenate([current,groups.get(d+1,np.array([],dtype=int))])
        pairs=cKDTree(points[ids]).query_pairs(chord(max(radii)), output_type='ndarray')
        if len(pairs):
            a,b=ids[pairs[:,0]],ids[pairs[:,1]]
            keep=(day[a]==d)|(day[b]==d)
            a,b=a[keep],b[keep]
            dt=np.abs(minutes[a]-minutes[b]); dist=arc(np.linalg.norm(points[a]-points[b],axis=1))
            for r in radii:
                spatial=dist<=r+1e-10
                for w in windows:
                    mask=spatial&(dt<=w)
                    np.add.at(counts[f'n_{r:g}km_{w}min'],a[mask],1)
                    np.add.at(counts[f'n_{r:g}km_{w}min'],b[mask],1)
            base=(dist<=1+1e-10)&(dt<=30)
            for key,mask in [('n_same_sat_1km_30min',base&(names[a]==names[b])),
                             ('n_other_sat_1km_30min',base&(names[a]!=names[b])),
                             ('n_different_time_1km_30min',base&(dt>0))]:
                np.add.at(counts[key],a[mask],1);np.add.at(counts[key],b[mask],1)
            m=base&(dt>0)
            later=np.where(minutes[a[m]]>minutes[b[m]],a[m],b[m])
            np.add.at(counts['n_prior_strict_1km_30min'],later,1)
        if progress and iteration%150==0: print(f'Neighbor calculation: day {iteration+1}/{len(groups)}',flush=True)
    result=pd.DataFrame(counts)
    result.insert(0,'sid',s.sid.to_numpy())
    return result

def regional_features(customer,sat,progress=False):
    # > 5 km halo here; largest tested radius is 2 km.
    bounds=customer_bbox(customer)
    regional=sat.loc[within_bbox(sat,customer_bbox(customer,0.05))].copy().reset_index(drop=True)
    regional['inside_region_bbox']=within_bbox(regional,bounds)
    points=xyz(customer.Latitude.to_numpy(),customer.Longitude.to_numpy())
    regional['nearest_customer_location_km']=arc(cKDTree(points).query(xyz(regional.latitude.to_numpy(),regional.longitude.to_numpy()))[0])
    features=neighbor_features(regional,progress=progress)
    return regional.merge(features,on='sid',validate='one_to_one')

def selection_metrics(target,prediction):
    target=np.asarray(target,bool); prediction=np.asarray(prediction,bool)
    tp=int((target&prediction).sum());fp=int((~target&prediction).sum())
    fn=int((target&~prediction).sum());tn=int((~target&~prediction).sum())
    def div(a,b): return a/b if b else np.nan
    return dict(listed_and_selected=tp,unlisted_selected=fp,listed_missed=fn,unlisted_not_selected=tn,
                selected=tp+fp,listed=tp+fn,listed_recall=div(tp,tp+fn),
                listed_precision_proxy=div(tp,tp+fp),listed_f1_proxy=div(2*tp,2*tp+fp+fn))

def candidate_rules(region):
    rules={}
    for r in [0.5,1,2]:
        for w in [0,10,30,60,1440]:
            for total in [2,3,5]:
                rules[f'r={r:g}km;t={w}min;n>={total}']=region[f'n_{r:g}km_{w}min'].ge(total-1).to_numpy()
    base=region['n_1km_30min'].ge(1).to_numpy()
    rules['quoted_same_satellite']=region.n_same_sat_1km_30min.ge(1).to_numpy()
    rules['quoted_other_satellite']=region.n_other_sat_1km_30min.ge(1).to_numpy()
    rules['quoted_different_time']=region.n_different_time_1km_30min.ge(1).to_numpy()
    rules['quoted_prior_strict']=region.n_prior_strict_1km_30min.ge(1).to_numpy()
    rules['all_hotspots']=np.ones(len(region),bool)
    for conf in [30,50,80]:
        rules[f'confidence>={conf}_only']=region.confidence_score.ge(conf).to_numpy()
    rules['type0_only']=region.type.eq(0).to_numpy()
    rules['day_only']=region.daynight.eq('D').to_numpy()
    for frp in [1,5,10,20,50]: rules[f'FRP>={frp}_only']=region.frp.ge(frp).to_numpy()
    quality=region.confidence_score.ge(50).to_numpy()
    # These are point-level post-filters; quality-filter-before-clustering is tested separately.
    rules['quoted + nominal_VIIRS_or_MODIS>=50']=base&quality
    rules['quoted + high_confidence_only']=base&region.confidence_score.ge(80).to_numpy()
    rules['quoted + type0']=base&region.type.eq(0).to_numpy()
    rules['quoted + day_only']=base&region.daynight.eq('D').to_numpy()
    rules['quoted + night_only']=base&region.daynight.eq('N').to_numpy()
    for frp in [1,5,10,20,50]: rules[f'quoted + FRP>={frp}']=base&region.frp.ge(frp).to_numpy()
    return rules

def synthetic_tests():
    # A-B-C chain crosses midnight. A-C exceeds both 1 km and 30 minutes.
    d=pd.DataFrame({'sid':range(5),'latitude':[0]*5,
        'longitude':np.array([0,.8,1.6,5,5.5])/EARTH_KM*180/np.pi,
        'utc':pd.to_datetime(['2019-01-01 23:50','2019-01-02 00:10','2019-01-02 00:30','2019-01-02 03:00','2019-01-02 03:00']),
        'satellite_name':['SNPP','SNPP','NOAA20','SNPP','NOAA20']})
    f=neighbor_features(d)
    assert f.n_1km_30min.tolist()==[1,2,1,1,1], f
    assert f.n_prior_strict_1km_30min.tolist()==[0,1,1,0,0]
    assert f.n_other_sat_1km_30min.tolist()==[0,1,1,1,1]
    assert f['n_1km_0min'].tolist()==[0,0,0,1,1]
    assert f['n_0.5km_30min'].tolist()==[0,0,0,1,1]
    assert neighbor_features(d.iloc[:1]).n_1km_30min.iloc[0]==0
    # Exactly 30 minutes is included; 31 minutes is excluded.
    edge=d.iloc[:2].copy();edge.utc=pd.to_datetime(['2019-01-01 00:00','2019-01-01 00:30'])
    assert neighbor_features(edge).n_1km_30min.tolist()==[1,1]
    edge.loc[edge.index[1],'utc']+=pd.Timedelta(minutes=1)
    assert neighbor_features(edge).n_1km_30min.tolist()==[0,0]
    pairs=pd.DataFrame({'delta_minutes':[420,361,360,450], 'same_satellite':[True]*4,
                        'unknown_satellite':[False]*4,'distance_km':[0]*4,
                        'customer_index':range(4),'sid':range(4)})
    assert select_matches(pairs).customer_index.tolist()==[0,1]
    # Compare the spatial/time engine against an independent brute-force oracle.
    rng=np.random.default_rng(12); n=60
    random=pd.DataFrame({'sid':range(n),'latitude':rng.uniform(-.02,.02,n),'longitude':rng.uniform(-.02,.02,n),
                         'utc':pd.Timestamp('2019-01-01 23:00')+pd.to_timedelta(rng.integers(0,240,n),unit='m'),
                         'satellite_name':rng.choice(['SNPP','NOAA20'],n)})
    f=neighbor_features(random)
    la=np.radians(random.latitude.to_numpy());lo=np.radians(random.longitude.to_numpy())
    hav=np.sin((la[:,None]-la)/2)**2+np.cos(la[:,None])*np.cos(la)*np.sin((lo[:,None]-lo)/2)**2
    distances=2*EARTH_KM*np.arcsin(np.sqrt(hav))
    times=random.utc.to_numpy().astype('datetime64[m]').astype(int)
    for r in [.5,1,2]:
        for w in [0,10,30,60,1440]:
            expected=((distances<=r)&(np.abs(times[:,None]-times)<=w)).sum(axis=1)-1
            assert np.array_equal(f[f'n_{r:g}km_{w}min'],expected)
    return 'Passed: midnight, chains, self-exclusion, spatial/time boundaries, simultaneous/prior/cross-satellite, matching hour bins, and brute-force neighbor comparisons.'

def fit_exploratory_trees(region):
    """Reconstruct list membership; unlisted examples are unknown, not true negatives.

    Fixed shallow trees, no hyperparameter selection using held-out labels.
    Within-year holdout consists of entire UTC dates, every fifth sorted date.
    Transfer fits are separate full-year training experiments.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    numeric=['confidence_score','frp','brightness_main','brightness_background','scan','track',
             'n_1km_30min','n_same_sat_1km_30min','n_other_sat_1km_30min','n_prior_strict_1km_30min']
    categoric=pd.get_dummies(region[['satellite_name','daynight']],dtype=float)
    physical=pd.concat([region[numeric],categoric],axis=1)
    geo=physical.assign(latitude=region.latitude,longitude=region.longitude,
                        local_hour=(region.utc+pd.Timedelta(hours=7)).dt.hour)
    models={'satellite_features':physical,'location_only':geo[['latitude','longitude']],
            'satellite_features_plus_location_hour':geo}
    results=[];texts={};importances=[]
    dates=region.utc.dt.normalize()
    for y in YEARS:
        iy=region.year.eq(y).to_numpy()
        test_dates=np.sort(dates.loc[iy].unique())[::5]
        test=iy&dates.isin(test_dates).to_numpy(); train=iy&~test
        for name,x in models.items():
            for experiment,train_mask,test_mask in [('heldout_dates',train,test),('other_year',iy,~iy)]:
                median=x.loc[train_mask].median().fillna(0)
                train_x=x.loc[train_mask].fillna(median)
                model=DecisionTreeClassifier(max_depth=4,min_samples_leaf=100,class_weight='balanced',random_state=42)
                model.fit(train_x,region.loc[train_mask,'listed'])
                pred=model.predict(x.loc[test_mask].fillna(median))
                results.append(dict(train_year=y,test_year=y if experiment=='heldout_dates' else YEARS[1-YEARS.index(y)],
                                    experiment=experiment,features=name,train_rows=int(train_mask.sum()),
                                    test_rows=int(test_mask.sum()),**selection_metrics(region.loc[test_mask,'listed'],pred)))
                key=f'{y}_{experiment}_{name}'
                texts[key]=export_text(model,feature_names=list(x.columns),decimals=3)
                for feature,value in zip(x.columns,model.feature_importances_):
                    importances.append(dict(model=key,feature=feature,importance=value))
    return pd.DataFrame(results),texts,pd.DataFrame(importances)
