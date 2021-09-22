from numpy import *
import pylab

_dir = '/home/enrico/SW/SNANA/sndata_root/filters/Bessell90/'

filt = 'UBVRI'
pylab.ion()
for f in filt:
    lam,tr = [],[]
    ff = open(_dir+f+'.dat')
    righe = ff.readlines()
    for r in righe:
        lam.append(float(r.split()[0]))
        tr.append(float(r.split()[1]))
    pylab.plot(lam,tr)

raw_input('...')
    
