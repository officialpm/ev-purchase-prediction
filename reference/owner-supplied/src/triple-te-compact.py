# Copyright 2026 Parth Maniar. Apache-2.0.
# Memory-bounded out-of-fold encoding test, no Kaggle upload.
import gc,time
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
P='/home/sandbox/s6e9/data/'
a=pd.read_csv(P+'train.csv');b=pd.read_csv(P+'test.csv');y=(a.Will_Buy_EV=='Yes').astype('int8').to_numpy();n=len(a)
cols=[c for c in b if c!='id'];raw=pd.concat([a[cols],b[cols]],ignore_index=True)
cat=[c for c in cols if not pd.api.types.is_numeric_dtype(raw[c])]
for c in cat:raw[c]=raw[c].astype('category')
for c in cols:raw[c+'_cnt']=raw[c].map(raw[c].value_counts()).astype('float32')
# Requested resolution: integer, floor(x/100), floor(x/1000).
K={}
for c in ['Annual_Income_USD','Daily_Commute_km']:
 v=pd.to_numeric(raw[c],errors='coerce').fillna(-999999).to_numpy()
 for d,label in [(1,'exact'),(100,'bin100'),(1000,'bin1000')]:K[c+'_'+label]=np.floor(v/d).astype('int32')
X=raw.iloc[:n].reset_index(drop=True);Xt=raw.iloc[n:].reset_index(drop=True);del raw,a,b;gc.collect()
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=1120)
# Encode outer-train with inner cross-fitting: never use own target, no validation labels.
def encode(k,y,tr,va,te,smooth,fold):
 def calc(train_idx,target_idx):
  df=pd.DataFrame({'key':k[train_idx],'y':y[train_idx]});g=df.groupby('key',sort=False).y.agg(['sum','count']);prior=float(y[train_idx].mean());
  if smooth=='auto':
   # Empirical-Bayes moment shrinkage per key. Stabilized variance denominator.
   p=g['sum']/g['count'];between=max(float(p.var()),1e-6);within=max(float(np.mean(p*(1-p))),1e-6);lam=max(1.0,within/between)
  else:lam=float(smooth)
  g['value']=(g['sum']+lam*prior)/(g['count']+lam)
  return pd.Series(k[target_idx]).map(g['value']).fillna(prior).to_numpy(dtype='float32')
 out=np.empty(len(tr),dtype='float32');inner=StratifiedKFold(n_splits=5,shuffle=True,random_state=1120+fold)
 for itr,iva in inner.split(tr,y[tr]):out[iva]=calc(tr[itr],tr[iva])
 return out,calc(tr,va),calc(tr,te)
O=np.zeros(n,dtype='float32');T=np.zeros(len(Xt),dtype='float32');start=time.time()
for fold,(tr,va) in enumerate(cv.split(X,y)):
 Ftr=X.iloc[tr].copy();Fva=X.iloc[va].copy();Ft=Xt.copy()
 for name,k in K.items():
  for smooth in ['auto',10.,100.]:
   q=encode(k,y,tr,va,np.arange(n,len(k)),smooth,fold);column=name+'_te_'+str(smooth)
   Ftr[column]=q[0];Fva[column]=q[1];Ft[column]=q[2]
 print('encoded',fold,'elapsed',round(time.time()-start),flush=True)
 m=lgb.LGBMClassifier(objective='binary',learning_rate=.03,n_estimators=4500,num_leaves=31,min_child_samples=60,colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_lambda=3,verbose=-1,n_jobs=3,random_state=1120)
 m.fit(Ftr,y[tr],eval_set=[(Fva,y[va])],eval_metric='auc',callbacks=[lgb.early_stopping(100,verbose=False)])
 O[va]=m.predict_proba(Fva)[:,1];T+=m.predict_proba(Ft)[:,1]/5
 print('fold',fold,'AUC',roc_auc_score(y[va],O[va]),'best',m.best_iteration_,'elapsed',round(time.time()-start),flush=True)
 np.save(P+'triple_te_compact_partial.npy',O)
 del Ftr,Fva,Ft,m;gc.collect()
print('SUMMARY triple_te_compact',roc_auc_score(y,O),'elapsed',round(time.time()-start),flush=True)
np.save(P+'triple_te_compact_oof.npy',O);np.save(P+'triple_te_compact_test.npy',T)
