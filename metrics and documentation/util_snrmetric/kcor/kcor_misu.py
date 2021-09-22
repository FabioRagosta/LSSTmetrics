description="compute k-correction (deredshifting spectra)"
"""########################################################################
2014/03/15 new kcor_misu version for input to fit_lc (v2.0)
"""########################################################################

import os,sys,shutil
import time
from numpy import *
import StringIO
import argparse
from pyraf import iraf
import pylab

sys.path.append('/home/enrico/scripts/alice')
import alice

iraf.onedspec(_doprint=0)
xx = iraf.stsdas(_doprint=0,Stdout=1)
iraf.hst_calib(_doprint=0)
iraf.synphot(_doprint=0)

asa_dir = '/home/supern/asa/'
#asa_dir = '/home/enrico/sne/sudare/kcor/'
alice_dir = '/home/supern/alice/data/'

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description=description,\
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("sn",help='sn name')
    parser.add_argument("-f", "--filt_in",dest="filt_in",default='BVR',
            type=str,help='filter in')
    parser.add_argument("-g", "--filt_ou",dest="filt_ou",default='gri',
            type=str,help='filter out')
    parser.add_argument("-z", "--zrange",dest="zrange",default='0.,1.,.05',
            type=str,help='redshift range and step')
    parser.add_argument("-v", "--verbose",dest="verbose",action="store_true",\
           default=False,help='Enable task progress report')

    args = parser.parse_args()


def assign_filter(g,filt_in,zrange):

    fpair,midf = [],[]
    for i in range(len(filt_in)-1):
        f1,f2 = filt_in[i],filt_in[i+1]
        lamf1 =  alice.bandpar[filt_in[i]][2]
        lamf2 =  alice.bandpar[filt_in[i+1]][2]
        fpair.append(f1+f2)
        midf.append((lamf1+lamf2)/2.)

    zra = {}
    for z in arange(zrange[0],zrange[1]+zrange[2],zrange[2]):
        f = filt_in[-1]
        for i in range(len(midf))[::-1]:
            lamg =  alice.bandpar[g][2]/(1+z)
            if lamg<midf[i]: f = fpair[i][0]
        if f not in zra.keys(): zra[f] = []
        zra[f].append(float('%5.2f' % z))

    return zra

def read_template_spectra(sn,jdmax):

    tspe = {}
    for c in ['spec','ph','wmin','wmax','flag']: tspe[c] = []

    ff = open('sn_template_spec.csv')
    righe = ff.readlines()
    for r in righe[4:]:
        _sn = r.split(',')[1]
        if sn==_sn:
            if sn=='Ia': ph = float(r.split(',')[2])
            else: ph = float(r.split(',')[2])-2400000-jdmax

            tspe['spec'].append(r.split(',')[0])
            tspe['ph'].append(ph)
            for i,c in enumerate(['wmin','wmax','flag']):
                tspe[c].append(float(r.split(',')[i+3].strip('\n')))
    return tspe

def kmmobs(i,filt_obs,filt_in,z0,wmin,wmax,verbose):
 
    err = StringIO.StringIO()
    if verbose: stderr = 0
    else: stderr = err

    lamc_obs,fwhm_obs = alice.bandpar[filt_obs][2],alice.bandpar[filt_obs][3]
    lamc_in,fwhm_in = alice.bandpar[filt_in][2],alice.bandpar[filt_in][3]

    pref = ''
    if wmin < lamc_in-fwhm_obs/2. and wmax > lamc_in+fwhm_obs/2.:
        if filt_obs in 'ugriz': pref = 'sdss,'
        _out =  iraf.calcphot(pref+filt_obs,'/tmp/_tmp.'+str(i),\
                'vegamag',Stdout=1,Stderr=stderr)
        mag_obs = float(_out[6].split()[1])
    else: mag_obs = 100

    if wmin/(1+z0) < lamc_in-fwhm_in/2. and wmax/(1+z0) > lamc_in+fwhm_in/2.:
        if filt_in in 'ugriz': pref = 'sdss,'
        vzero = str(1/(1+z0)-1)
        _out =  iraf.calcphot(pref+filt_in,'z(/tmp/_tmp.'+str(i)+',$0)',\
              'vegamag',vzero=vzero,Stdout=1,Stderr=stderr)
        mag_in = float(_out[6].split()[1])
    else: mag_in = 100

    if mag_obs !=100 and mag_in != 100: k0obs = mag_in-mag_obs
    else: k0obs = 100

    return k0obs

def kmm(i,filt_in,filt_ou,zra,z0,wmin,wmax,verbose):
 
    err = StringIO.StringIO()
    if verbose: stderr = 0
    else: stderr = err

    lamc_in,fwhm_in = alice.bandpar[filt_in][2],alice.bandpar[filt_in][3]
    lamc_ou,fwhm_ou = alice.bandpar[filt_ou][2],alice.bandpar[filt_ou][3]

    pref = ''
    if wmin/(1+z0) < lamc_in-fwhm_in/2. and wmax/(1+z0) > lamc_in+fwhm_in/2.:
        if filt_in in 'ugriz': pref = 'sdss,'
        vzero = str(1/(1+z0)-1)
        _out =  iraf.calcphot(pref+filt_in,'z(/tmp/_tmp.'+str(i)+',$0)',\
              'vegamag',vzero=vzero,Stdout=1,Stderr=stderr)
        mag_in = float(_out[6].split()[1])
    else: mag_in = 100

    pref = ''
    if filt_ou in 'ugriz': pref='sdss,'
    vzero = []
    for z in zra:
        zz = z-z0
        if zz <0: vzero.append('%5.4f' % (1/(1+zz)-1))
        else: vzero.append('%5.4f' % zz)

    vzero = ','.join(vzero)

    _out =  iraf.calcphot(pref+filt_ou,'z(/tmp/_tmp.'+str(i)+',$0)','abmag',\
          vzero=vzero,Stdout=1,Stderr=stderr)
    kcor = []
    for i,m in enumerate(_out[6:]):
        if  wmin*(1+zra[i]-z0) < lamc_ou-fwhm_ou/2. and \
              wmax*(1+zra[i]-z0) > lamc_ou+fwhm_ou/2. and \
              float(m.split()[1]) != 100 and mag_in != 100:
                kcor.append(float(m.split()[1])-mag_in) 
        else: kcor.append(100)
    return kcor

def kcor_compute(sn,tspe,snlist,filt_in,filt_ou,zrange,verbose): #########

    AB = snlist['AB']
    zed = snlist['zed']
    lam = arange(3000,10000,1)
    ablam = array([alice.cardelli(l,3.1) for l in lam])

    kphobs,k0obs,kph,kz,kcor = [],{},{},{},{}

    for i,spe in enumerate(tspe['spec']):
        if verbose: print '   ',spe
        iraf.delete('/tmp/_tmp?'+str(i),verify=False)
        iraf.wspec(asa_dir+sn+"/"+spe+'[*,1,1]',"/tmp/_tmp."+str(i),\
                       header=False)

        ff = open("/tmp/_tmp."+str(i))
        righe = ff.readlines()
        ll,fl = [],[]
        for r in righe: 
            if tspe['wmin'][i]<float(r.split()[0])<tspe['wmax'][i]: 
                ll.append(float(r.split()[0]))
                fl.append(float(r.split()[1]))

        if AB>0.2:  
            ll,_fl = array(ll),array(fl)
            iablam = interp(ll,lam,ablam)
            fl = _fl*10**(0.4*iablam*AB*.75)    # .75 convert AB to AV

        ff = open('/tmp/_tmp.'+str(i),'w')
        for l in range(len(ll)): ff.write(str(ll[l])+' '+str(fl[l])+' \n')
        ff.close()
        
        kphobs.append(tspe['ph'][i])
        for f in filt_in:  ## SHIFT FILTER TO MATCH ZED
            if zed < .15: fobs = f
            else: 
                ii =  filt_in.index(f)
                if ii+1 < len(filt_in): fobs = filt_in[filt_in.index(f)+1]
                else: fobs =f 
            if fobs+f not in k0obs.keys(): k0obs[fobs+f] = []
    
            k0obs[fobs+f].append(kmmobs(i,f,fobs,zed,tspe['wmin'][i],\
                                     tspe['wmax'][i],False))

        for g in filt_ou:
            zra = assign_filter(g,filt_in,zrange)
            for f in zra.keys():
                if verbose: print '-',g+f,'-  zrange=',zra[f]
                kmmout = kmm(i,f,g,zra[f],zed,tspe['wmin'][i],\
                                     tspe['wmax'][i],False)    
                if kmmout:
                    if any(array(kmmout)!=100):
                        if g+f not in kcor.keys():
                            kph[g+f],kz[g+f],kcor[g+f] = [],[],[]
                        kcor[g+f].append(kmmout)
                        kph[g+f].append(tspe['ph'][i])
                        kz[g+f] = zra[f]

    for gf in kph.keys():     
        kph[gf],kcor[gf] = array(kph[gf]),array(kcor[gf])

    for gf in kph.keys():                    #     clean output
        for i in range(len(kz[gf]))[::-1]:   # delete elements from the end
            if all(array(kcor[gf])[:,i]==100):
                kz[gf] = delete(kz[gf],i)
                kcor[gf] = delete(kcor[gf],i,1)
                
    return kphobs,k0obs,kph,kz,kcor

###########################################################################
if __name__ == "__main__":

    sne = args.sn.split(',')
    zrange = [float(x) for x in args.zrange.split(',')]
    filt_in = [x for x in args.filt_in]
    filt_ou = [x for x in args.filt_ou]

    ff = open('sn_template.list')
    righe = ff.readlines()
    snlist={}
    for r in righe:
        if r[0]=='#': continue
        _sn = r.split()[0]
        snlist[_sn] = {}
        snlist[_sn]['type'] = r.split()[1]
        zed = float(r.split()[2])
        if abs(zed)>1: zed *= 1/300000.        

        snlist[_sn]['zed'] = zed
  
        snlist[_sn]['jdmax'] = float(r.split()[4])
        snlist[_sn]['AB'] = float(r.split()[6])+float(r.split()[7])

    if sne=='all': sne = snlist.keys()

    for sn in sne:
                                                 # read template spectra 
        print 10*"*"+"   ",sn
        tspe = read_template_spectra(sn,snlist[sn]['jdmax'])  
        if os.path.exists('kk/'+sn+'.kcor'): 
            print '>>> WARNING: file ',sn+'.kcor','already exists'
            sys.exit()

        kphobs,k0obs,kph,kz,kcor = kcor_compute(sn,tspe,snlist[sn],\
                         filt_in,filt_ou,zrange,args.verbose) 

        ff = open('kk/'+sn+'.kcor','w')
        ff.write('z= %5.3f \n'% snlist[sn]['zed'])
        for j in range(len(kphobs)): 
            ff.write('%6.1f ' % kphobs[j])
        ff.write('\n')
        for ffobs in k0obs.keys():
            ff.write(ffobs+' *** \n')
            for j in range(len(k0obs[ffobs])): 
                ff.write('%6.2f ' % k0obs[ffobs][j])
            ff.write('\n')
            
        for gf in kph.keys():
            ff.write(gf+' \n')
            ff.write('--- ')
            for j in range(len(kph[gf])): 
                ff.write('%6.1f ' % kph[gf][j])
            ff.write('\n')

            for i,zed in enumerate(kz[gf]):
                    ff.write('%5.2f ' % zed)
                    for j in range(len(kcor[gf][:,i])):
                        ff.write('%6.2f ' % kcor[gf][j,i])
                    ff.write('\n')
        ff.close()

    print "********** Completed in ",int(time.time()-start_time),"sec"

