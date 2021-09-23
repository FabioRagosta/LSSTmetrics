import sys, os, h5py, pickle, warnings, subprocess, time, re
sys.path.append('/home/idies/workspace/Storage/fragosta/persistent/LSST_OpSim/Scripts_NBs')
from opsimUtils import *
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
from util_snrmetric import cosmo
from lsst.sims.maf.metrics import BaseMetric
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots
from lsst.sims.utils import equatorialFromGalactic, galacticFromEquatorial
from lsst.sims.maf.utils import m52snr, radec2pix
from itertools import groupby
os.environ['SNANA_DIR']='/home/idies/workspace/Storage/fragosta/persistent/LSST_OpSim/Scripts_NBs/SNRate_Simulations/SNANA'
os.environ['SNDATA_ROOT']='/home/idies/workspace/Storage/fragosta/persistent/LSST_OpSim/Scripts_NBs/SNRate_Simulations/SNDATA_ROOT'
os.environ['LSST_DIR']='./util_snrmetric'
kcor_dir= './util_snrmetric/'
metric_dir = './'

__all__=['template_lc','getDataMetric','SNclassification_metric']



class template_lc:
    def get_filter(self):

        ff = open(kcor_dir+'filters.csv')
        righe = ff.readlines()
        band_label = righe[0].strip('\n').split(',') 

        bandpar = {}
        for r in righe[1:]:
            _r = r.strip('\n').split(',')
            if len(_r)<=1 : continue
            bandpar[_r[0]] = {}
            for b in band_label:
                i = band_label.index(b)
                if b in ['lameff','bandwidth','zeropoint','abmvega']:
                    bandpar[_r[0]][b] = float(_r[i])
                    if b in ['lameff','bandwidth']: bandpar[_r[0]][b] *= 10.
                else: bandpar[_r[0]][b] = _r[i]

        return band_label,bandpar
    def __init__(self,metricName='template_lc', sn_group={'Ia':{'Ia':(['1990N','1992A','1994D','2002bo'],70),  
                'IaBright':(['1991T','1999ee'],10), 
                'IaFaint':(['1991bg'],15), 
                'Iapec':(['2000cx','2002cx'],5)},
          'II':{'IIP':(['1999em','2004et','2009bw'],60),        # 60
                'IIFaint':(['1999br','1999gi','2005cs'],10),    # 10
                'IIL':(['1992H'],10),                           # 10
                'IIb':(['1993J','2008ax'],10),                  # 10
                'II87A':(['1987A'],10)},                        # 10
          'IIn':{'IIn':(['2010jl'],45),
                 'IIna':(['1998S'],45), 
                 'IIpec':(['1997cy','2005gj'],10)},
          'Ibc':{'Ib':(['2009jf','2008D'],27), 
                 'Ic':(['1994I','2004aw','2007gr'],68), 
                 'IcBL':(['1998bw'],5)}, 
          'SLSN' : {'SLSN':(['2008es'],100)}}, z_min=0,z_max=1.,z_step=0.1, extinction=0, path='$HOME', dataout=False):
        self.sngroup =sn_group
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step
        self.extinction = extinction
        self.path = path
        self.dataout = dataout
        self.filtri='gri'
        self.zrange = np.arange(self.z_min,self.z_max,self.z_step)
        self.bandpar={}
        self.band_label= ['bandw','fwhm','avgwv','equvw','zpoint','abmag for vega']
        #landolt & johnson buser and kurucz 78
        self.bandpar['U']=[205.79,484.6,3652,542.62,4.327e-9,  0.76] 
        self.bandpar['B']=[352.94,831.11,4448,1010.3,6.09e-9, -0.11] 
        self.bandpar['V']=[351.01,826.57,5505,870.65,3.53e-9, 0.] 
        #landolt & cousin bessel 83
        self.bandpar['R']=[589.71,1388.7,6555,1452.2,2.104e-9, 0.18]
        self.bandpar['I']=[381.67,898.77,7900.4,1226,1.157e-9, 0.42]
        # HAWK-I filter
        self.bandpar['Y']=[0,1019.4,10226.9,0,5.74e-10, '?'] 
        #bessel in bessel and brett 88
        self.bandpar['J']=[747.1,1759.3,12370,2034,3.05e-10, '?'] 
        self.bandpar['H']=[866.55,2040.6,16471,2882.8,1.11e-10, '?'] 
        self.bandpar['K']=[1188.9,2799.6,22126,3664.3,3.83e-11, '?']
        # ASIAGO PHOTOMETRIC DATABASE
        self.bandpar['L']=[0.,9000.,34000,0.,8.1e-12, '?'] 
        self.bandpar['M']=[0.,11000.,50000,0.,2.2e-12, '?'] 
        self.bandpar['N']=[0.,60000.,102000,0.,1.23e-13, '?']
        # sloan
        self.bandpar['u']=[194.41,457.79, 3561.8, 60.587,3.622e-9, 0.92]
        self.bandpar['g']=[394.17,928.19, 4718.9, 418.52,5.414e-9, -0.11]
        self.bandpar['r']=[344.67,811.65, 6185.2, 546.14,2.472e-9, 0.14]
        self.bandpar['i']=[379.57,893.82, 7499.8, 442.15,1.383e-9, 0.36]
        self.bandpar['z']=[502.45,1183.2, 8961.5, 88.505,8.15e-10, 0.52] 
        # SWIFT A-> UW1, D --> UM2, S -> UW2 
        #self.bandpar['A']=[348.43,820.5,2650.6,770.45,3.818e-9] 
        #self.bandpar['D']=[189.46,446.14,2269.2,519.03,4.321e-9]
        #self.bandpar['S']=[285.12,671.4,2136.7,639.45,4.825e-9] 
        self.bandpar['A']=[0.,693.,2634,0.,4.3e-9, '?'] # Poole et al. 2008 383, 627
        self.bandpar['D']=[0.,498.,2231,0.,4.0e-9, '?'] # corrected for z-point
        self.bandpar['S']=[0.,657.,2030,0.,5.2e-9, '?']
        self.snlist= []
        for ty in self.sngroup:
            for sty in self.sngroup[ty]:
                self.snlist += self.sngroup[ty][sty][0]
    
        self.b_label, self.bpar= self.get_filter()
        #########################    LAST FORMAT
    def lastform(self, riga):
        sndata = {}
        sndata['sn'] = str.split(riga[0])[0]
        sndata['sntype'] = str.split(riga[0])[1]
        sndata['galaxy'] = str.split(riga[1])[0]
        par = ['ABg','ABi','mu','Rv']
        for _par in par:
            _ip = str.find(str.lower(riga[1]),str.lower(_par))
            if _ip >=0:
                sndata[_par]     = float(riga[1][_ip+1:].split()[1])
                sndata[_par+'_err'] = float(riga[1][_ip+1:].split()[2])

        sndata['jd_expl'] = False
        if 'jd_expl=' in riga[0]:
            _ir = str.index(riga[0],'=')+1
            sndata['jd_expl'] = float(str.split(riga[0][_ir:])[0])

        jdmax,magmax,jdmax_err,magmax_err = {},{},{},{}

        iband = []
        for i in range(2,len(riga)):
            _r = str.strip(riga[i])
            if _r:
                if _r[0] in 'ugriz' and  'jd_max' in str.lower(_r):
                    iband.append(i)
                if _r[0] in 'YJHKLMN' and  'jd_max' in str.lower(_r):
                    iband.append(i)
                if _r[0] in 'ASD' and  'jd_max' in str.lower(_r):
                    iband.append(i)
        iband.append(len(riga))

        riri = [[3,iband[0]-1]]
        for i in range(1,len(iband)):
            riri.append([riri[i-1][1]+1,iband[i]-1])

        bands = ''
        jd,mag,mag_err,source ={},{},{},{}
        for ri in riri:
            _bands = str.split(riga[ri[0]])[0]
            bands += _bands
            for b in _bands:
                c = str.index(_bands,b)*2+2
                jdmax[b]     = float(str.split(riga[ri[0]])[c])
                jdmax_err[b] = float(str.split(riga[ri[0]])[c+1])
                magmax[b]    = float(str.split(riga[ri[0]+1])[c-1])
                magmax_err[b]= float(str.split(riga[ri[0]+1])[c])
                jd[b],mag[b],mag_err[b],source[b] = [],[],[],[]
                for i in range(ri[0]+3,ri[1]):
                    if len(str.strip(riga[i])) > 0:
                        if str.lstrip(riga[i])[0] != '#':
                            _mag = float(str.split(riga[i])[c])
                            if _mag<999:
                                jd[b].append(float(str.split(riga[i])[1]))
                                mag[b].append(float(str.split(riga[i])[c]))
                                mag_err[b].append(float(str.split(riga[i])[c+1]))
                                source[b].append(float(str.split(riga[i])[len(_bands)*2+2]))
         # bands sort
        lam = []
        for b in bands:
            lam.append(self.bandpar[b][self.band_label.index('avgwv')])
        _bands = ''
        for i in np.argsort(lam):
            _bands += bands[i]

        sndata['bands'] = _bands
        sndata['jdmax'],sndata['jdmax_err']=jdmax,jdmax_err
        sndata['magmax'],sndata['magmax_err'] = magmax,magmax_err
        sndata['format'] = 'LAST'
        sndata['jd'],sndata['source']=jd,source
        sndata['mag'],sndata['mag_err'] = mag,mag_err

        return sndata
##################################  NEW FORM    
    def newform(self,riga):

        sndata = {}
        sndata['sn'] = str.split(riga[0])[0]
        sndata['sntype'] = str.split(riga[1])[0]
        sndata['galaxy'] = str.split(riga[3])[0]
        par = ['ABg','ABi','mu','Rv']
        for _par in par:
            _ip = str.find(riga[3],_par)
            if _ip >=0: 
                sndata[_par]     = float(riga[3][_ip+1:].split()[1])
                sndata[_par+'_err'] = float(riga[3][_ip+1:].split()[2])

        sndata['jd_expl'] = False
        if 'jd_expl=' in riga[0]:
            _ir = str.index(riga[0],'=')+1
            sndata['jd_expl'] = float(str.split(riga[0][_ir:])[0])

        jdmax,magmax,jdmax_err,magmax_err = {},{},{},{}
        bands = str.split(riga[4])[0]
        jd,mag,mag_err,source ={},{},{},{}
        for b in bands: 
            c = str.index(bands,b)*2+1
            jdmax[b]     = float(str.split(riga[0])[c])
            jdmax_err[b] = float(str.split(riga[0])[c+1])
            magmax[b]    = float(str.split(riga[1])[c])
            magmax_err[b]= float(str.split(riga[1])[c+1])
            c += 1
            jd[b],mag[b],mag_err[b],source[b] = [],[],[],[]
            for i in range(6,len(riga)):
                if len(str.strip(riga[i])) > 0:
                    if str.lstrip(riga[i])[0] != '#':
                        _mag = float(str.split(riga[i])[c])
                        if _mag < 999:
                            jd[b].append(float(str.split(riga[i])[1]))
                            mag[b].append(_mag)
                            mag_err[b].append(float(str.split(riga[i])[c+1]))
                            source[b].append(float(str.split(riga[i])[len(bands)*2+2]))

        sndata['bands'] = bands
        sndata['jdmax'],sndata['jdmax_err']=jdmax,jdmax_err
        sndata['magmax'],sndata['magmax_err'] = magmax,magmax_err 
        sndata['format'] = 'NEW'
        sndata['jd'],sndata['source']=jd,source
        sndata['mag'],sndata['mag_err'] = mag,mag_err

        return sndata

    ################################   OLD FORM
    def oldform(self, riga):
        sn      = str.split(riga[0])[0]
        sntype  = str.split(riga[1])[3]
        galaxy  = str.split(riga[0])[3]
        abg     = float(str.split(riga[0])[4])
        abg_err = 0.
        abi     = float(str.split(riga[1])[4])
        abi_err = 0.
        mu      = float(str.split(riga[0])[5])
        mu_err  = 0. 
        jd_expl = False
        if 'jd_expl=' in riga[0]:
            _ir = str.index(riga[0],'=')+1
            sndata['jd_expl'] = float(str.split(riga[0][_ir:])[0])

        jdmax,magmax,jdmax_err,magmax_err = {},{},{},{}
        jdmax['B']     = float(str.split(riga[0])[1])
        jdmax_err['B'] = 0.
        magmax['B']    = float(str.split(riga[0])[2])
        magmax_err['B']= 0.

        _bands = str.split(riga[3])[3:-1]
        _jd,_mag,_source = {},{},{}
        for b in _bands:
            _jd[b],_mag[b],_source[b] = [],[],[]
        for i in range(4,len(riga)):
            for c in range(len(_bands)):
                b =  _bands[c]
                if len(str.strip(riga[i])) > 0:
                    if str.lstrip(riga[i])[0] != '#':
                        if str.find(str.split(riga[i])[c+3],':') > 0:
                            _mm = float(str.split(riga[i])[c+3][:-1])
                        else:
                            _mm = float(str.split(riga[i])[c+3])
                        if _mm > 0:
                            _jd[b].append(float(str.split(riga[i])[2]))
                            _mag[b].append(_mm)
                            _str = str.split(riga[i])[len(_bands)+3]
                            try: _strf = float(_str)
                            except:
                                if 'lim' in _str: _strf = -1
                                else: _strf = 1
                            _source[b].append(_strf)

        for b in _bands:
            if 'lim' in b:
                if b[:-3] in _bands:
                    _jd[b[:-3]] +=_jd[b]
                    _mag[b[:-3]] += _mag[b]
                    _source[b[:-3]] += (zeros(len(_source[b]))-1).tolist()
                else:
                    _jd[b[:-3]] =_jd[b]
                    _mag[b[:-3]] = _mag[b]
                    _source[b[:-3]] = (zeros(len(_source[b]))-1).tolist()
                    _bands += b[:-3]

        for b in _bands:
            if b == 'Mpg':
                if 'B' in _bands:
                    _jd['B'] += _jd[b]
                    _mag['B'] += (array(_mag[b])+0.29).tolist()
                    _source['B'] += _source[b]
                else:    
                    _jd['B'] = _jd[b]
                    _mag['B'] = (array(_mag[b])+0.29).tolist()
                    _source['B'] = _source[b]
                _bands += 'B'
            if b == 'Mpv' or b == 'vis':
                if 'V' in _bands:
                    _jd['V'] += _jd[b]
                    _mag['V'] += _mag[b]
                    _source['V'] += _source[b]
                else:
                    _jd['V'] = _jd[b]
                    _mag['V'] = _mag[b]
                    _source['V'] = _source[b]
                _bands += 'V'
            if b == 'CCD':
                if 'R' in _bands:
                    _jd['R'] += _jd[b]
                    _mag['R'] += _mag[b]
                    _source['R'] += _source[b]
                else:
                    _jd['R'] = _jd[b]
                    _mag['R'] = _mag[b]
                    _source['R'] = _source[b]
                _bands += 'R'

        bands =''
        jd,mag,mag_err,source = {},{},{},{}
        for b in _bands:
             if b in ['U','B','V','R','I','J','H','K']:
                bands += b
                jdmax[b]     =  jdmax['B']
                jdmax_err[b] =  jdmax_err['B']
                magmax[b]    =  magmax['B']
                magmax_err[b]=  magmax_err['B']
                jd[b] = _jd[b]                                      
                mag[b] = _mag[b]
                mag_err[b] = (zeros(len(mag[b]))).tolist()
                source[b] = _source[b]

        sndata = {'sn':sn, 'sntype':sntype, 'galaxy':galaxy, 'bands':bands,
                   'ABg':abg, 'ABg_err':abg_err, 'ABi':abi, 'ABi_err':abi_err,\
                   'mu':mu, 'mu_err':mu_err, 'jd_expl':jd_expl, \
                   'jdmax':jdmax, 'jdmax_err':jdmax_err, 'magmax':magmax,\
                   'magmax_err':magmax_err, 'format':'OLD',\
                   'jd':jd, 'mag':mag, 'mag_err':mag_err, 'source':source}
        return sndata


    def leggifile(self, snfile):
        lcf = open(snfile+'.dat','r')
        riga=lcf.readlines()
        check1 = str.split(riga[0])[1]
        check2 = str.split(riga[1])[0]

        if re.search('[a-zA-Z]',check1):         
            return self.lastform(riga)
        elif re.search('[a-zA-Z]',check2):
            #print snfile,'NEW',
            return self.newform(riga)            
        else:
            #print snfile,'OLD',
            return self.oldform(riga)

    def cardelli(self, lam,Rv):
    # CARDELLI LAW (Cardelli et al. 1989, ApJ 345, 245)
      x = 10000./lam
      # CARDELLI start from 0.3, but NED extends it to L band 
      if x>=0.1 and x<=1.1:
          y = x**1.61
          a = 0.574*y
          b = -0.527*y

      if x>1.1 and x <3.3:
          y = x-1.82 
          a = 1+y*(0.17699 + y * (-0.50447 + y * (-0.02427 +y * (0.72085 + y * (0.01979 + y * (-0.77530 + y * 0.32999))))))
          b = y * (1.41338 + y * (2.28305 + y * (1.07233 + y * (-5.38434 + y * (-0.62251 + y * (5.30260 +y * (-2.09002)))))))

      if x>=3.3 and x <8.0:
          y = (x - 4.67)**2 
          a = 1.752 - 0.316 * x - 0.104 / (y + 0.341)
          b = -3.090 + 1.825 *x + 1.206 / (y + 0.263) 

          if x>=5.9 and x<8.0:
              y = x - 5.9
              a +=  - 0.04473 * y**2 - 0.009779 * y**3
              b +=  + 0.2130 * y**2 + 0.1207 * y**3

      if x>=8.0:	
          y = x - 8. 
          a = -1.073 - 0.628 * y + 0.137 * y**2 - 0.070 * y**3
          b = 13.670 + 4.257 * y - 0.420 * y**2 + 0.374 * y**3


      Al_AV = a + b/ Rv
      return Al_AV

    def AX(self, band='',AB='',R_V=3.1):

        if not band: band = raw_input('<<  Band ? ')
        if AB=='': AB = float(raw_input('<<  Ab absorption ? '))

        abfac = self.cardelli(self.bandpar[band][self.band_label.index('avgwv')],R_V)\
                /self.cardelli(self.bandpar['B'][self.band_label.index('avgwv')],R_V)
        return AB*abfac


    def read_template_lc(self, sndata,tsn):  #  read candidate light curves

        if all(np.array([x  in sndata['bands'] for x in 'BVR'])): _bands,b = 'BVR','B'
        elif all(np.array([x  in sndata['bands'] for x in 'gri'])):_bands,b = 'gri','g'

        jdmax = sndata['jdmax'][b]
        Rv = 3.1
        if sndata['ABi'] > 99:
            print ('!!! WARNING: for SN',_sn,'ABi=',str(sndata['ABi']),\
                       ' (not available) ==> set to 0')
            sndata['ABi'] = 0.0

        ph,absmag = {},{}

        for b in _bands:
            if b not in sndata['bands']:
                print ("!!! ERROR: band",b,"not available for SN",sndata['sn'])
                sys.exit()
            abx = self.AX(b,sndata['ABg']+sndata['ABi'],R_V=Rv)
            jd,mag,source = np.array(sndata['jd'][b]),np.array(sndata['mag'][b]),\
                np.array(sndata['source'][b])
            __ph = jd-jdmax

            ii =np.where((source>=0)&(__ph<175))
            _ph,_absmag = __ph[ii],mag[ii]-tsn['mu']-abx
            jj = np.argsort(_ph)
            ph[b],absmag[b] = _ph[jj],_absmag[jj]

        EBV = (sndata['ABg']+sndata['ABi'])/4.1

        return ph,absmag,jdmax,sndata['bands'],EBV

    def kcor_read(self, sn,tsn,): #########

        kph,kz,kcor,kiko = {},{},{},{}

        ff = open(kcor_dir+'/kcor/kk/'+sn+'.kcor')
        righe = ff.readlines()
        kkph = [float(x) for x in righe[1].split()]

        ign = []       # read correction for specific reshift template
        for i,r in enumerate(righe):
            if '***' in r: ign.append(i)
        for i in ign:
            gn = righe[i].split()[0]
            kiko[gn] = np.array([float(k) for k in righe[i+1].split()])

        ff = open(kcor_dir+'/kcor/kk/'+tsn['kkclass']+'.kk')
        righe = ff.readlines()

        igf = []                               #   read kcor for SN class
        for i,r in enumerate(righe):
            if '---' in r: igf.append(i-1)

        for n,i in enumerate(igf):
            gf = righe[i].split()[0]
            kz[gf],kcor[gf] = [],[]
            i1 = len(righe)
            if n<len(igf)-1: i1 = igf[n+1]
            kph[gf] = np.array([float(p) for p in righe[i+1].split()[1:]])

            for r in righe[i+2:i1]:
                kz[gf].append(float(r.split()[0]))
                kcor[gf].append(np.array([float(k) for k in r.split()[1:]]))
        ff.close()

        return kkph,kiko,kph,kz,kcor

    def kcor_interpolate(self, tph,z,kph,kz,kcor):

        if len(kz)>1: 
            jz = np.searchsorted(kz,z)-1
            if jz < len(kz)-1:
                il = np.where(kcor[jz]<100)
                _kl = np.interp(tph,kph[il],kcor[jz][il])
                iu = np.where(kcor[jz+1]<100)
                _ku = np.interp(tph,kph[iu],kcor[jz+1][iu])    
                _kcor = _kl+(z-kz[jz])*(_ku-_kl)/(kz[jz+1]-kz[jz])
            else:
                il = np.where(kcor[jz]<100)
                _kcor = np.interp(tph,kph[il],kcor[jz][il])
        else:
            il = np.where(kcor[0]<100)
            _kcor = np.interp(tph,kph[il],kcor[0][il])

        return _kcor


    def mag_observer_frame(self, tph,tabsmag,tzed,z,kcor,kiko,tEXT):

        phobs = tph*(1-tzed+float(z))
        magobs = tabsmag+kcor+kiko+cosmo.mu(float(z))+tEXT
        return phobs,magobs

    def plot_template_lc(phobs,magobs,label):

        pylab.title(label)
        pylab.plot(phobs,magobs,'ob')
        pylab.ylim(max(magobs)+.5,min(magobs)-.5)
        pylab.show()

    def read_template_list(self):

        #####
        print('I\'m reading the templates')
        ff = open(kcor_dir+'/kcor/sn_template.list')          # read template list/info
        righe = ff.readlines()
        tsn = {}
        _alltypes = []
        for r in righe:
            if r[0]=='#': continue
            _sn = r.split()[0]
            tsn[_sn] = {}
            tsn[_sn]['type'] = r.split()[1]
            if '-' in tsn[_sn]['type']: tsn[_sn]['kkclass'] = _sn
            else: tsn[_sn]['kkclass'] = tsn[_sn]['type']
            tsn[_sn]['zed'] = float(r.split()[2])
            if abs(tsn[_sn]['zed'])>1: tsn[_sn]['zed'] *= 1/300000.
            tsn[_sn]['mu'] = float(r.split()[3])
            _alltypes.append(tsn[_sn]['type'])

        return tsn,set(_alltypes)

# read template light curves
    def read_shift_template(self, sn,tsn,alltypes):

        sndata = self.leggifile(kcor_dir+'/kcor/lc/'+sn)
        tph,tabsmag,tsn[sn]['jdmax'],tsn[sn]['bands'],tsn[sn]['EBV'] =\
                                                                       self.read_template_lc(sndata,tsn[sn])

        #print ("    SN=",sndata['sn'],tsn[sn]['bands'])

        kkph,kiko,kph,kz,kcor = {},{},{},{},{}     #  read k-correction

        #print ('compute Kcor ',sn)
        kkph,kiko,kph,kz,kcor = self.kcor_read(sn,tsn[sn]) 

        phobs,magobs = {},{}
        for z in self.zrange:
            phobs[z],magobs[z] ={},{}
            _z = z-.05
            for g in self.filtri:
                for _k in kz:
                    if _k[0]==g:
                        if  z >= min(kz[_k])-.05:
                            f = _k[1]
                for de in kiko:
                    d,e = de
                    if e == f:  fin = d

                #print (sn,g,f,z)
                _kcor = self.kcor_interpolate(tph[fin],_z,kph[g+f],kz[g+f],kcor[g+f])
                bb = np.where(kiko[fin+f]<100)
                _kiko = np.interp(tph[fin],np.array(kkph)[bb],kiko[fin+f][bb])
                avfact = self.cardelli(self.bpar[g]['lameff'],3.1)

                _phobs,_magobs = self.mag_observer_frame(tph[fin],tabsmag[fin],
                                    tsn[sn]['zed'],_z,_kcor,_kiko,avfact*self.extinction)

    #            phobs[z][ff] = arange(min(_phobs),max(_phobs))
                phobs[z][g] = np.arange(min(_phobs),max(_phobs))
                magobs[z][g]  = np.interp(phobs[z][g] ,_phobs,_magobs)

    #            plot_template_lc(phobs[ff][z],magobs[ff][z],sn+'  '+ff+'  '+str(z))


        return phobs,magobs

    def run(self):
        tsn,alltypes = self.read_template_list()
        phobs,magobs = {},{}
        for sn in self.snlist:
            phobs[sn],magobs[sn] = self.read_shift_template(sn,tsn,alltypes)
        if self.dataout:
            if self.path :
                dir_path= self.path
            else:
                dir_path = '$HOME'
            ff = open(dir_path,'w')
            for sn in self.snlist:
                ff.write('*** '+sn+' \n')
                for z in self.zrange:
                    ff.write('** {:.1f} \n'.format(z))
                    for f in self.filtri:
                        ff.write('* '+f+' \n')
                        for p,m in zip(phobs[sn][z][f],magobs[sn][z][f]):
                            ff.write('{} {} \n'.format(p,m))

            return print('A file was created with all the LC\'s templates in the z_range setted. The file is in the directory: {}'.format(dir_path))
        else:
            return{'phobs': phobs, 'magobs': magobs}

class getDataMetric(metrics.BaseMetric):
    """
    extract data from database by column names, 
    combined with UniSlicer()
    """
    
    def __init__(self, colname=['expMJD', 'airmass'], **kwargs):
        self.colname = colname
        super().__init__(col=colname, metricDtype='object', **kwargs)
        
        
    def run(self, dataSlice, slicePoint=None):
        
        # return dataSlice directly
        result = dataSlice
        
        return result





class SNclassification_metric_DDF(BaseMetric):
    """
    Parameters:
    ___________
    Survey Parameters:
        mjdCol= MJD observations column name from Opsim database      (DEFAULT = observationStartMJD) 
        m5Col= Magnitude limit column name from Opsim database      (DEFAULT = fiveSigmaDepth)
        filterCol= Filters column name from Opsim database      (DEFAULT = filter)
        exptimeCol = Column name for the total exposure time of the visit(DEFAULT = visitExposureTime)
        nightCol = The night's column of the survey (starting at 1) (DEFAULT = night)
        vistimeCol = Column name for the total time of the visit (DEFAULT = visitTime)
        RACol= RA column name from Opsim database      (DEFAULT = fieldRA)
        DecCol= Dec column name from Opsim database      (DEFAULT = fieldDec)
        surveyDuration= Survey Duration      (DEFAULT = 10)
        surveyStart= Survey start date      (DEFAULT = None)
        
   
    Template parameters:
        templates= dictionary with all the SNe templates     (DEFAULT= {'Ia':{'Ia':(['1990N','1992A','1994D','2002bo'],100)}})
        z = array with z min, z max and z step for the redshift range     (DEFAULT= [0.1,1,0.1])
        explosiontime = times of explosion array      (DEFAULT= None)
        
    
    Detection parameters:
        detectSNR= dictionary with SNR threshold for each filter    (DEFAULT = {'u': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5, 'y': 5})
        nFilters= None or list of filters constrained for the classification threshold  (DEFAULT = None)
        ndetect= integer, threshold number of points detected on the lightcurve (aside from the filters)   (DEFAULT = 3)
        
    Classification parameters
        nclass= integer, threshold number of points detected on the lightcurve (aside from the filters) to be classified  (DEFAULT = 3)
    __________
    
    OUTPUT:
        dataout = True, Dictionary containing the coordinates of all the SNe detected, the time of explosions and the number of 
        detected and no-detected along with the number of classifies, un-classified and mis-classified SNe for each type
        
        dataout = False, fraction of correctly classified SNe
        
    """
    def __init__(self, metricName='SNclassification_metric_DDF', mjdCol='observationStartMJD', 
                 RACol='fieldRA', DecCol='fieldDec',filterCol='filter', m5Col='fiveSigmaDepth', 
                 exptimeCol='visitExposureTime',nightCol='night',vistimeCol='visitTime',
                 surveyDuration=10.,surveyStart=None,observedFilter=['g','r','i'],
                 templates= {'Ia':{'Ia':(['1990N','1992A','1994D','2002bo'],100)}},
                 z=[0.1,1,0.1], explosiontime=None, 
                 detectSNR={'u': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5, 'y': 5},
                 nFilters=None, ndetect=3, nclass=5, dataout=False, **kwargs):
        
        "Survey Parameters"
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.nightCol = nightCol
        self.vistimeCol = vistimeCol
        self.observedFilter = observedFilter
        self.surveyDuration = surveyDuration
        self.surveyStart = surveyStart 
        
        "Parameters to generate magnitude for a template lc at different redshits and explosion times"
        self.templates=templates
        self.z = z
        self.explosiontime = explosiontime  
        
        "Parametes to contrains the detections' selection"
        self.nFilters = nFilters
        self.ndetect = ndetect
        self.detectSNR = detectSNR
        
        "Parametes to contrains the classification"
        self.nclass = nclass
        
        "Paremeters to select the kind of output"
        self.dataout = dataout
        
        
        # if you want to get the light curve in output you need to define the metricDtype as object
        if self.dataout:
            super(SNclassification_metric_DDF, self).__init__(col=[self.mjdCol,self.m5Col, self.filterCol,self.RACol,
                                                                   self.DecCol,self.exptimeCol,self.nightCol,
                                                                   self.vistimeCol],
                                                       metricDtype='object', units='',
                                                       metricName=metricName, **kwargs)
        else:
            super(SNclassification_metric_DDF, self).__init__(col=[self.mjdCol, self.m5Col, self.filterCol],
                                                       units='Fraction Detected', metricName=metricName,
                                                       **kwargs)
        
        
        
        
        
        # The SN template light curves are simulated at the redshifts in the redshift range  
        # K-correction is applied at each redshift z 
        if not os.path.exists('./template'):
            os.makedirs('./template')
        zmin = self.z[0]
        zmax = self.z[1]
        zstep = self.z[2]
        temp = template_lc.template_lc(sn_group= self.templates, z_min=zmin,z_max= zmax,z_step=zstep)
        self.obs_template = temp.run()
        self.zrange = temp.zrange
        self.filtri = temp.filtri
        for j, z in enumerate(self.zrange):
                for ty in self.templates:       
                    for sty in self.templates[ty]:
                        for sn in self.templates[ty][sty][0]:
                            asciifile = './template/snlc_{}_z={}_DDF.ascii'.format(sn,str(z))
                            ff = open(asciifile,'w')
                            if ty in ['Ia','Ibc']:    endTime = 50.*(1+z)
                            else:                     endTime =100.*(1+z)
                            for f in self.filtri:
                                for i,p in enumerate(self.obs_template['phobs'][sn][z][f]):
                                    if self.obs_template['phobs'][sn][z][f][i] > endTime: break    
                                    ff.write('{:.2f} {:.3f} {}\n'.format(p,self.obs_template['magobs'][sn][z][f][i],f))
                            ff.close()
                        
    def read_lightCurve(self, asciifile):
        """Reads in an ascii file, from the simulated ligh curves, 3 columns: epoch, magnitude, filter

        Returns
        -------
        numpy.ndarray
            The data read from the ascii text file, in a numpy structured array with columns
            'ph' (phase / epoch, in days), 'mag' (magnitude), 'flt' (filter for the magnitude).
        """
        if not os.path.isfile(asciifile):
            raise IOError('Could not find lightcurve ascii file %s' % (asciifile))
        self.lcv_template = np.genfromtxt(asciifile, dtype=[('ph', 'f8'), ('mag', 'f8'), ('flt', 'S1')])
        self.transDuration = self.lcv_template['ph'].max() - self.lcv_template['ph'].min()

    def make_lightCurve(self, time, filters):
        """Turn lightcurve definition into magnitudes at a series of times.

        Parameters
        ----------
        time : numpy.ndarray
            The times of the observations.
        filters : numpy.ndarray
            The filters of the observations.

        Returns
        -------
        numpy.ndarray
             The magnitudes of the transient at the times and in the filters of the observations.
        """
        lcMags = np.zeros(time.size, dtype=float)
        for key in set(self.lcv_template['flt']):
            fMatch_ascii = np.where(np.array(self.lcv_template['flt']) == key)[0]
            # Interpolate the lightcurve template to the times of the observations, in this filter.
            temp_ph=np.array(self.lcv_template['ph'], float)[fMatch_ascii]-np.array(self.lcv_template['ph'], float)[fMatch_ascii][0]
            lc_ascii_filter = np.interp(time, temp_ph,
                                        np.array(self.lcv_template['mag'], float)[fMatch_ascii])
            lcMags[filters == key.decode("utf-8")] = lc_ascii_filter[filters == key.decode("utf-8")]
        return lcMags
    
    def coadd(self, data):
        """
        Method to coadd data per band and per night
        Parameters
        ------------
        data : `pd.DataFrame`
            pandas df of observations
        Returns
        -------
        coadded data : `pd.DataFrame`
        """

        keygroup = [self.filterCol, self.nightCol]

        data.sort_values(by=keygroup, ascending=[
                         True, True], inplace=True)

        coadd_df = data.groupby(keygroup).agg({self.vistimeCol: ['sum'],
                                               self.exptimeCol: ['sum'],
                                               self.mjdCol: ['mean'],
                                               self.RACol: ['mean'],
                                               self.DecCol: ['mean'],
                                               self.m5Col: ['mean']}).reset_index()

        coadd_df.columns = [self.filterCol, self.nightCol, 
                            self.vistimeCol, self.exptimeCol, self.mjdCol,
                            self.RACol, self.DecCol, self.m5Col]

        coadd_df.loc[:, self.m5Col] += 1.25 * \
            np.log10(coadd_df[self.vistimeCol]/30.)

        coadd_df.sort_values(by=[self.filterCol, self.nightCol], ascending=[
                             True, True], inplace=True)

        return coadd_df.to_records(index=False)
    def custom_split(self,x='',c='',index=0):
        x=x.decode("utf-8")
        return x.split(c)[index]
    def run(self, dataSlice, slicePoint=None): 
        # Sort the entire dataSlice in order of time.
        dataSlice.sort(order=self.mjdCol)
        print(dataSlice)
        dataSlice = self.coadd(pd.DataFrame(dataSlice))
        print(dataSlice)
        # Check that surveyDuration is not larger than the time of observations we obtained.
        # (if it is, then the nTransMax will not be accurate).
        tSpan = (dataSlice[self.mjdCol].max() - dataSlice[self.mjdCol].min()) / 365.25
        surveyDuration = np.max([tSpan, self.surveyDuration])
        
        lc = {}
        filterNames = self.observedFilter
        filterN= ''.join(filterNames)
        fieldRA = np.median(dataSlice['fieldRA']) 
        fieldDec = np.median(dataSlice['fieldDec'])
        
        
        expldist=[]
        # We set the survey duration if it is not given as input
        if self.surveyStart is None:
            surveyStart = dataSlice[self.mjdCol].min()
        else:
            surveyStart = dataSlice[self.mjdCol].min() + 365*self.surveyStart
        
        """
        the array of the explosion dates has three possible setting:
            self.explosiontime = integer,  it sets the number of dates to drawn randomly from the Opsim database
            self.explosiontime = None, one date is randomly drawn from the Opsim database
            self.explosiontime = list or array, it sets the array containing the explosion dates
        """
        if isinstance(self.explosiontime, int):
            expl_t =np.random.choice(dataSlice[self.mjdCol],self.explosiontime)
        elif  self.explosiontime is None:
            expl_t=np.random.choice(dataSlice[self.mjdCol],1)
        elif isinstance(self.explosiontime, (list, tuple, np.ndarray)):
            expl_t=np.array(self.explosiontime)+surveyStart
                
        if all(np.in1d(self.observedFilter, dataSlice[self.filterCol])): #check if all the filters for the observed lightcurves are available
            
            index_filter = np.in1d(dataSlice[self.filterCol], self.observedFilter) 
            obs_filter = dataSlice[self.filterCol][index_filter]  
            obs = dataSlice[self.mjdCol][index_filter]           
            obs_m5 = dataSlice[self.m5Col][index_filter]
            
            classify =pd.DataFrame(index=self.zrange,columns=['pixId', 'Ia','Ibc','II','UNKNOWN','nDet','nClassified'])
            classify['pixId']=radec2pix(16,np.radians(fieldRA),np.radians(fieldDec))           
            CM ={}
            
            expldist={}
            temp_list = glob.glob("./template/*.ascii")
            for t in temp_list:
                """
                Observed light curve simulation step:
                """
                
                expldist=[]
                sn_list = 0 # index to count the simulated SNe setted to zero
                nDetected = 0 # index to count the detected SNe setted to zero
                nUnDetected = 0 # index to count the no-detected SNe setted to zero  
                classifiable = 0
                unclassifiable = 0
                listout=[]
                sn, z = t.split('_')[1],t.split('_')[2].split('=')[1]
                print([t,sn,z])
                Ia=['1990N','1992A','1994D','2002bo','1991T','1999ee','1991bg','2000cx','2002cx']
                Ibc=['2009jf','2008D','1994I','2004aw','2007gr','1998bw']
                II=['1999em','2004et','2009bw','1999br','1999gi','2005cs','1992H','1993J','2008ax','1987A','2010jl','1998S','1997cy','2005gj','2008es']
                if sn in Ia:
                    ty='Ia'
                if sn in Ibc:
                    ty='Ibc'
                if sn in II:
                    ty='II'
                self.read_lightCurve(t) # we read the simulated lightcurve at the given z
                for k,times in enumerate(expl_t):
                    sn_list+=1
                    expldist.append(times) 
                    indexlc = np.where((obs>= times) & (obs<=times+self.transDuration)) # we create a mask for all the observation whitin the transient duration
                    lcEpoch = (obs[indexlc] - times) # define the dates of the phases from the explosion time 

                    if np.size(indexlc)>0: 
                        lcMags = self.make_lightCurve(lcEpoch, obs_filter[indexlc]) # Generate the observed light curve magnitudes
                        lcSNR = m52snr(lcMags, obs_m5[indexlc])
                        lcpoints_AboveThresh = np.zeros(len(lcSNR), dtype=bool) 
                        nfilt_det = []
                        nfilt_class = []
                        for f in self.observedFilter:                    
                            filtermatch = np.where(obs_filter[indexlc] == f)
                            lcpoints_AboveThresh[filtermatch] = np.where(lcSNR[filtermatch] >= self.detectSNR[f],True,False) # we define a mask for the detected points on the light curve     
                        Dpoints = np.sum(lcpoints_AboveThresh) #counts the number of detected points
                        if Dpoints>=self.ndetect: 
                            nDetected+=1
                        else:
                            nUnDetected+=1

                        if self.nFilters:
                            Dpoints =0
                            nfilt_class=[]
                            for f in self.nFilters:                    
                                filtermatch = np.where(obs_filter[indexlc] == f)
                                epoch_filt=lcEpoch[filtermatch]
                                mask_class = epoch_filt<30
                                if any(mask_class):
                                    Dpoint_class += np.sum(lcpoints_AboveThresh[filtermatch][mask_class])
                                else:
                                    Dpoint_class += 0
                            if Dpoints>=self.nclass: nfilt_class.append(True)
                        else:
                            
                            mask_class = lcEpoch<30
                            if any(mask_class):
                                
                                Dpoint_class = np.sum(lcpoints_AboveThresh[mask_class])
                            else:
                                
                                Dpoint_class = 0
                            
                            
                        
                        snname='LSST_{}_{}_{}_DDF.dat'.format(sn,z,np.round(times,2))
                        if self.nFilters: 
                            if np.sum(nfilt_class) == np.size(self.nFilters):
                                listout.append(snname+'\n')

                        else:
                            if Dpoint_class>=self.nclass: 
                                classifiable += 1
                                listout.append(snname+'\n')
                            else:
                                unclassifiable +=1
                                

                        if snname+'\n' in listout:
                            lc = {}
                            lc["Mags"] = lcMags
                            lc["filter"] = obs_filter[indexlc]
                            lc["SNR"] = lcSNR
                            lc["Epoch"] = obs[indexlc]
                            lc['detect'] = lcpoints_AboveThresh
                            # producing a file to pass to PSNID for the classification

                            mag = {}
                            jd = {}
                            merr = {}
                            snr={}

                            output  = 'SURVEY:  LSST \n'                
                            output += 'SNID: {}_{} \n'.format(sn,k)
                            output += 'IAUC:    UNKNOWN \n'             
                            output +=  'RA:     '+str(fieldRA)+'  deg \n'
                            output +=  'DECL:   '+str(fieldDec)+'  deg \n'
                            output +=  'MWEBV:    0.0  MW E(B-V) \n'                           
                            output +=  'REDSHIFT_FINAL:  '+z+' +- '+'%5.3f' % self.z[2]+' (CMB)\n'
                            output +=  'FILTERS:  {}   \n'.format(filterN)               
                            output +=  ' \n'
                            output += '# ======================================\n' 
                            output += '# TERSE LIGHT CURVE OUTPUT\n' 
                            output += '#\n' 
                            output += 'NOBS: {} \n'.format(np.size(lcMags[lcpoints_AboveThresh])) 
                            output += 'NVAR: 8 \n'
                            output += 'VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR   SNR    MAG     MAGERR \n'
                            for f in filterNames:
                                filtermatch = np.where(obs_filter[indexlc] == f)
                                detect= np.array(lc['detect'][filtermatch])
                                mag[f] = lcMags[filtermatch][detect]
                                jd[f] = obs[indexlc][filtermatch][detect]
                                snr[f] = lcSNR[filtermatch][detect]
                                merr[f] = 2.5*np.log10(1+1/snr[f])
                                for h,j in enumerate(jd[f]):

                                    fl = 10**(-0.4*(mag[f][h]))*1e11
                                    if snr[f][h]>1: 

                                        flerr = fl/snr[f][h]/1.3
                                    else: 

                                        flerr = fl/1.1
                                    output += 'OBS: %9.3f   %s NULL  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f \n' % (j,f,fl,flerr,snr[f][h],mag[f][h],merr[f][h])
                            output +='END: '


                            ofile = open(os.path.join('./LC_DDF','LSST_{}_{}_{}_DDF.dat'.format(sn,z,np.round(times,2))),'w')
                            ofile.write(output)
                            ofile.close()

                    else:
                        nUnDetected += 1

                listsn=open(os.path.join('./LC_DDF','LSST.LIST'),'w')
                listsn.writelines(listout)
                listsn.close()
                if not os.path.exists(os.path.join('./LC_DDF','LSST.README')):
                    dumbfile=open(os.path.join('./LC_DDF','LSST.README'),'w')
                    dumbfile.close()
                if not os.path.exists(os.path.join('./LC_DDF','LSST.IGNORE')):
                    dumbfile=open(os.path.join('./LC_DDF','LSST.IGNORE'),'w')
                    dumbfile.close()    
                
                
                """
                Classification step
                ____________________
                
                We run PSNID on the detected light curves, the PSNID output is saved as a string array in the variable r
                """
                classify['nDet'][float(z)]= [nDetected, nUnDetected]
                print(len(listout))
                if len(listout) >0:
                    start_time_class = time.time()
                    r = subprocess.check_output([os.environ['SNANA_DIR']+'/bin/psnid.exe', os.environ['LSST_DIR']+'/PSNID_LSST_DD.nml'], stderr=subprocess.STDOUT)
                    
                    
                    CM[float(z)]= pd.DataFrame(columns= ['Ia','Ibc','II'], index= ['Ia','Ibc','II'])
                    classify['Ia'][float(z)]=0
                    classify['Ibc'][float(z)]=0
                    classify['II'][float(z)]=0
                    classify['UNKNOWN'][float(z)]=0
                    CM[float(z)]['Ia']['Ia']=0
                    CM[float(z)]['Ibc']['Ia']=0
                    CM[float(z)]['II']['Ia']=0
                    CM[float(z)]['Ia']['Ibc']=0
                    CM[float(z)]['Ibc']['Ibc']=0
                    CM[float(z)]['II']['Ibc']=0
                    CM[float(z)]['Ia']['II']=0
                    CM[float(z)]['Ibc']['II']=0
                    CM[float(z)]['II']['II']=0
                    nClassified=0
                    
                    

                    # we search for classification flags in the variable r 
                    line= np.array(r.split())
                    custom_split = np.vectorize(self.custom_split)
                    types = np.where(line==b'type')
                    sn_t = custom_split(x=line[np.where(line==b'Done')[0]+3],c='_',index=0)
                    sn_Ia= np.in1d(sn_t,Ia)
                    sn_Ibc= np.in1d(sn_t,Ibc)
                    sn_II= np.in1d(sn_t,II)
                    
                    classify['Ia'][float(z)]+=np.nansum(line[types[0]+2][sn_Ia]==b'Ia')
                    classify['Ibc'][float(z)]+=np.nansum(line[types[0]+2][sn_Ibc]==b'Ibc')
                    classify['II'][float(z)]+=np.nansum(line[types[0]+2][sn_II]==b'II')
                    classify['UNKNOWN'][float(z)]+=np.nansum(line[types[0]+2]==b'UNKNOWN')
                    nClassified+= classify['Ia'][float(z)]+classify['Ibc'][float(z)]+classify['II'][float(z)]
                    nUnClassified = len(listout)-nClassified                         
                    # confusion matrix
                    CM[float(z)]['Ia']['Ia']+=np.nansum(line[types[0]+2][sn_Ia]==b'Ia')/(np.nansum(sn_Ia)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['Ibc']['Ia']+=np.nansum(line[types[0]+2][sn_Ia]==b'Ibc')/(np.nansum(sn_Ia)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['II']['Ia']+=np.nansum(line[types[0]+2][sn_Ia]==b'II')/(np.nansum(sn_Ia)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['Ia']['Ibc']+=np.nansum(line[types[0]+2][sn_Ibc]==b'Ia')/(np.nansum(sn_Ibc)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['Ibc']['Ibc']+=np.nansum(line[types[0]+2][sn_Ibc]==b'Ibc')/(np.nansum(sn_Ibc)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['II']['Ibc']+=np.nansum(line[types[0]+2][sn_Ibc]==b'II')/(np.nansum(sn_Ibc)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['Ia']['II']+=np.nansum(line[types[0]+2][sn_II]==b'Ia')/(np.nansum(sn_II)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['Ibc']['II']+=np.nansum(line[types[0]+2][sn_II]==b'Ibc')/(np.nansum(sn_II)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    CM[float(z)]['II']['II']+=np.nansum(line[types[0]+2][sn_II]==b'II')/(np.nansum(sn_II)-np.nansum(line[types[0]+2]==b'UNKNOWN'))
                    print('________________________________________')
                    
                    print('\n at z ={} for ty= {}, {}-like: Ia={},Ibc={},II={} '.format(z, ty, sn, classify['Ia'][float(z)],classify['Ibc'][float(z)],classify['II'][float(z)]))
                    
                    classify['nClassified'][float(z)]=[('UnClassified',nUnClassified),('classified',nClassified)]
                    print('total SNe simulated: {}'.format(sn_list))
                    print('detected:{}, nonDetected:{},classifiable:{}, unclassifiable:{}, classificate:{}, nonClassificate:{}, UNKNWON:{}'.format(nDetected, 
                                                                                                    nUnDetected,classifiable, unclassifiable-nUnDetected,
                                                                                                    nClassified,nUnClassified,classify['UNKNOWN'][float(z)])) 
                    CM[float(z)]=CM[float(z)].fillna(0)
                    print(CM[float(z)])
                    print("\n --- {:.2f} minutes ---\n".format((float(time.time()) - float(start_time_class))/60))
                else:
                    print('\n at z ={} for ty= {}: no classifiable lc '.format(z, ty))
            if self.dataout:
                explosiontime= np.array(expldist)
                return {'ConfusionMetric': CM,'class':classify,'time_expl':expl_t,'SN_coo':[fieldRA,fieldDec]}

            else:
                N=np.sum([classify[['Ia','Ibc','II']][z] for z in self.zrange])/len(listout)            
                return float(N)
            


