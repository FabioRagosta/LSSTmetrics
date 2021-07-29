from numpy import *
from pyraf import iraf
import sys

ff = open('/home/enrico/SW/SNANA/sndata_root/snsed/Hsiao07.dat')
righe = ff.readlines()

phi,lam,fl = [],[],[]
for r in righe:
    phi.append(float(r.split()[0]))
    lam.append(float(r.split()[1]))   
    fl.append(float(r.split()[2]))

ph = set(phi)

phi,lam,fl = array(phi),array(lam),array(fl)

for p in ph:
    ff = open('tmp.asc','w')
    ii = where(phi==p)
    for i in ii[0]:
        ff.write('%9.2f %14.7g \n' % (lam[i],fl[i]))
    iraf.rspectext('tmp.asc','Ia_'+str(p),crval1=1000,cdelt1=10)
    iraf.hedit('Ia_'+str(p),'IDENT',p,add=True,verify=False)
