from __future__ import annotations
import math
import numpy as np
from phase7_adaptive_dev import jsd_matrix, norm, EPS

def prior_localcorr(E,M,power=2.,q=3,min_support=2):
 E=norm(E);M=np.asarray(M,bool);C,D=E.shape;J=jsd_matrix(E)
 sig=[]
 for i in range(C): sig.append(float(np.median(np.sort(J[i,np.arange(C)!=i]))))
 sig=np.asarray(sig);G=np.eye(C)
 for i in range(C):
  for k in range(i+1,C):
   gs=math.sqrt(max(sig[i],EPS)*max(sig[k],EPS));r=J[i,k]/max(gs,EPS);G[i,k]=G[k,i]=math.exp(-(r**power))
 P=np.zeros_like(E); peer_mass=[]; missing_import=[]
 for i in range(C):
  peers=np.array([k for k in np.argsort(-G[i]) if k!=i][:min(q,C-1)],int)
  num_all=np.zeros(D);den_all=np.zeros(D)
  for k in range(C):
   if k==i:continue
   w=G[i,k]*M[k];num_all+=w*E[k];den_all+=w
  p=np.divide(num_all,den_all,out=np.zeros_like(num_all),where=den_all>EPS)
  miss=~M[i]
  allow=np.zeros(D,bool)
  if len(peers): allow=(M[peers].sum(0)>=min_support)
  block=miss&(~allow);p[block]=E[i,block]
  # For allowed missing coords, use only the local corroboration peers, not distant peers.
  allowed_missing=miss&allow
  if np.any(allowed_missing):
   num=np.zeros(D);den=np.zeros(D)
   for k in peers:
    w=G[i,k]*M[k];num+=w*E[k];den+=w
   p[allowed_missing]=np.divide(num,den,out=np.zeros_like(num),where=den>EPS)[allowed_missing]
  p[den_all<=EPS]=E[i,den_all<=EPS]
  if p.sum()<=EPS:p=E[i].copy()
  P[i]=p/max(p.sum(),EPS);peer_mass.append(G[i].sum()-1);missing_import.append(float(P[i,miss].sum()) if np.any(miss) else 0)
 return P,{'peer_mass_mean':float(np.mean(peer_mass)),'missing_import_mass':float(np.mean(missing_import))}

