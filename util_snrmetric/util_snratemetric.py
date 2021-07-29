####### utilities for SNRate metric ######
####### in this module you can find:
# class template_lc: produce k-corrected template lc 
# class transientsmetric: produce observed LC from a template
# class phot_class: produce a photometric classification performed using SNANA
# func singleplot: produce a plot of the observed LC from a template

from builtins import zip
import os, pickle, glob
import matplotlib.pyplot as plt
from util_snrmetric import cosmo
import numpy as np
import pandas as pd
from lsst.sims.maf.metrics import BaseMetric
from lsst.sims.maf.utils import m52snr
import subprocess
from subprocess import Popen, PIPE
from pylab import cm
from lsst.sims.maf.utils.mafUtils import radec2pix
kcor_dir= '/home/idies/workspace/Storage/fragosta/persistent/LSST_OpSim/Scripts_NBs/SNRate_Simulations/util_snrmetric/'
metric_dir = '/home/idies/workspace/Storage/fragosta/persistent/LSST_OpSim/Scripts_NBs'
######## class dedicated to the simulate the K-corrected LC's templates at different redshift:
####  sn_group: dict of the template type, subtype and fraction of each template per subgroup;
####  z_min, z_max, z_step: floats to define the range and the step in redshift to perform the simulation;
####  path: string array with the path where to save the .txt file with the LCs; if no path is passed to the class the $HOME path is used intead.
####  extinction: float for the value of the extinction, default = 0
####  dataout: boolean, if True it is created a file "template_lc.txt" with the kcorrected magnitude and phases of the template in the selected  filters and redshift  range; if False the output is a dictionary with the kcorrected magnitude and phases for each redshift bin and filter. (Default dataout=False)
import re

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

                print (sn,g,f,z)
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


class transientsmetric(BaseMetric):
    def __init__(self, z=[0.1,1,0.1], metricName='transientsmetric', filename=None, mjdCol='expMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', RACol='fieldRA', DecCol='fieldDec', seeingCol='seeingFwhmEff',observedFilter=['g','r','i'],
                 surveyDuration=10.,surveyStart=None, explosiontime=None,templates= {'Ia':{'Ia':(['1990N','1992A','1994D','2002bo'],100)}}, detectSNR={'u': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5, 'y': 5}, nFilters=1, npoints=3, label = None,
                 dataout=False,classify_ty=False,classify_zrange=False,ty=None, **kwargs):
        
        "templates"
        self.templates=templates
        "Parameters to generate magnitude for a template lc at different redshits and explosion times"
        self.z = z
        self.ty = ty
        self.explosiontime = explosiontime
        "Survey Parameters"
        self.observedFilter= observedFilter
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.RACol = RACol
        self.DecCol=DecCol
        self.filterCol = filterCol
        self.seeingCol = seeingCol
        self.surveyDuration = surveyDuration
        self.surveyStart = surveyStart 
        "Paremeters to select the kind of output"
        self.dataout = dataout
        self.classify_ty = classify_ty
        self.classify_zrange = classify_zrange
        self.filename=filename
        "Parametes to contrains the detections' selection"
        self.nFilters = nFilters
        self.npoints = npoints
        self.detectSNR = detectSNR
        
        # if you want to get the light curve in output you need to define the metricDtype as object
        if self.dataout:
            super(transientsmetric, self).__init__(col=[self.mjdCol, self.m5Col, self.filterCol,self.RACol,self.DecCol,self.seeingCol],
                                                       metricDtype='object', units='',
                                                       metricName=metricName, **kwargs)
        else:
            super(transientsmetric, self).__init__(col=[self.mjdCol, self.m5Col, self.filterCol],
                                                       units='Fraction Detected', metricName=metricName,
                                                       **kwargs)
        
        
        
        
    
        # Read ascii lightcurve template here. It doesn't change per slicePoint.
        #self.read_lightCurve(asciifile)
        zmin = self.z[0]
        zmax = self.z[1]
        zstep = self.z[2]
        temp = template_lc(sn_group= self.templates, z_min=zmin,z_max= zmax,z_step=zstep)
        self.obs_template = temp.run()
        self.zrange = temp.zrange
        self.filtri = temp.filtri
        
        
    def save_to_file(self, dic, filename):
        '''save dict item to csv file'''
        
        df = pd.DataFrame(dic)
        with open(filename, 'a') as f:
            df.to_csv(f, header=f.tell()==0, index=None)
            
    def read_lightCurve(self, asciifile):
        """Reads in an ascii file, 3 columns: epoch, magnitude, filter

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
             The magnitudes of the object at the times and in the filters of the observations.
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

    def run(self, dataSlice, slicePoint=None):
        # Sort the entire dataSlice in order of time.
        dataSlice.sort(order=self.mjdCol)
        # Check that surveyDuration is not larger than the time of observations we obtained.
        # (if it is, then the nTransMax will not be accurate).
        tSpan = (dataSlice[self.mjdCol].max() - dataSlice[self.mjdCol].min()) / 365.25
        surveyDuration = np.max([tSpan, self.surveyDuration])

        if self.surveyStart is None:
            surveyStart = dataSlice[self.mjdCol].min()
        else:
            surveyStart = dataSlice[self.mjdCol].min() + 365*self.surveyStart
        
        #########self.expMJD =  surveyStart
                    # Set up the atmospheric condition for the observations
        """ If we want to reproduce different atmosferic condition form what are simulated within the DB,
        we can pass to the metric through slicePoint different seeing condition and estimate different 
        fiveSigmalimits
        """
        try:
            m5= dataSlice[self.m5Col]+2.5*np.log10(dataSlice[self.seeingCol]/slicePoint['seeing'])
        except:
            m5= dataSlice[self.m5Col]
        
        if isinstance(self.explosiontime, int):
            expl_t =np.random.choice(dataSlice[self.mjdCol],self.explosiontime)
        elif  self.explosiontime is None:
            expl_t=np.random.choice(dataSlice[self.mjdCol],1)
        elif isinstance(self.explosiontime, (list, tuple, np.ndarray)):
            expl_t=np.array(self.explosiontime)+surveyStart
            
        
        lc = {}
        filterNames = self.observedFilter
        filterN= ''.join(filterNames)
        
        if self.classify_zrange:
            classify ={z: [0,0,0] for z in self.zrange}
            fieldRA = np.mean(dataSlice['fieldRA']) 
            fieldDec = np.mean(dataSlice['fieldDec'])
            classify.update({'pixId':radec2pix(16,np.radians(fieldRA),np.radians(fieldDec))})
        
        sngroups= self.templates
        expldist=[]
        #check if all the filters for the observed lightcurves are available
        if all(np.in1d(self.observedFilter, dataSlice[self.filterCol])):
            index_filter = np.in1d( dataSlice[self.filterCol], self.observedFilter)
            obs_filter = dataSlice[self.filterCol][index_filter]
            obs = dataSlice[self.mjdCol][index_filter]        
            obs_m5 = m5[index_filter]
            nTransMax = 0            
            for j, z in enumerate(self.zrange):
                sn_list = 0
                nDetected = 0
                nNoDetected = 0
                if (self.classify_zrange):    
                        listout=open(os.path.join('./LC','LSST.LIST'),'w')
                for ty in sngroups:
                    lc[ty]={}       
                    for sty in sngroups[ty]:
                        lc[ty][sty]={}
                        for sn in sngroups[ty][sty][0]:
                            lc[ty][sty][sn]={}                            
                            ff = open('snlc.ascii','w')
                            if ty in ['Ia','Ibc']:    endTime = 50.*(1+z)
                            else:                     endTime =100.*(1+z)
                            for f in self.filtri:
                                for i,p in enumerate(self.obs_template['phobs'][sn][z][f]):
                                    if self.obs_template['phobs'][sn][z][f][i] > endTime: break    
                                    ff.write('{:.2f} {:.3f} {}\n'.format(p,self.obs_template['magobs'][sn][z][f][i],f))
                            ff.close()
                            asciifile = 'snlc.ascii'
                            zz= [sn,  'z='+str(z)]
                        
                            self.read_lightCurve(asciifile)
                          
        #index_pointing = np.isclose(dataSlice[self.RACol],ra,atol=1e-1) & np.isclose(dataSlice[self.DecCol],dec,atol=1e-1) 
        
                            for k,time in enumerate(expl_t):
                                expldist.append(time)
            # Calculate the time/epoch for each lightcurve.
                                indexlc = np.where((obs>= time) & (obs<=time+self.transDuration))
                                lcEpoch = (obs[indexlc] - surveyStart ) % self.transDuration
                                if np.size(indexlc)>0:   
                                    """lcNumber is an array of integers and it is evaluated dividing the SurveyDuration by the Transient Duration, 
                                    thus different lcNumber identitfy detection of a supernova that explode in different epoch during the survey
                                    """
                            # Total number of transient which have reached detection threshholds.
                                
                            # Total number of transients which could possibly be detected,
                            # given survey duration and transient duration.
                                    nTransMax += np.ceil(surveyDuration / (self.transDuration / 365.25))
                


                # Generate the actual light curve magnitudes and SNR
                                
                                    lcMags = self.make_lightCurve(lcEpoch, obs_filter[indexlc])
                                    dm=[]
                                    bin_t=[]
                                    lcSNR = m52snr(lcMags, obs_m5[indexlc])
                #lc[time]['MJD'] = obs[indexlc]               
                                    lcpoints_AboveThresh = np.zeros(len(lcSNR), dtype=bool)
                                    for f in np.unique(dataSlice[self.filterCol][indexlc]):                    
                                            filtermatch = np.where(obs_filter[indexlc] == f)[0]
                                            lcpoints_AboveThresh[filtermatch] = np.where(lcSNR[filtermatch] >= self.detectSNR[f],True,False)
                                    if len(lcpoints_AboveThresh) > 3:
                                        lc[ty][sty][sn][time] = {}
                                        lc[ty][sty][sn][time]["Mags"] = lcMags
                                        lc[ty][sty][sn][time]["filter"] = obs_filter[indexlc]
                                        lc[ty][sty][sn][time]["SNR"] = lcSNR
                                        lc[ty][sty][sn][time]["Epoch"] = lcEpoch
                                        lc[ty][sty][sn][time]['detect'] = lcpoints_AboveThresh
                                        for f in np.unique(dataSlice[self.filterCol][indexlc]): 
                                            fil = np.where(obs_filter[indexlc]==f)[0]
                                            filtermatch = np.where(obs_filter[indexlc] == f)[0]
                                            l = lcMags[fil][lcpoints_AboveThresh[filtermatch]]
                                            dm.append([l[i+1]-l[i] for i in range(len(l)-1)])
                                        val_t, vv = np.histogram(lcEpoch[lcpoints_AboveThresh],10) 
                                        if len(np.where(val_t>0)[0])>5:
                                            bin_t.append(True)
                                # produce a file to pass to snana for the classification
                                        if (self.classify_ty) or (self.classify_zrange):
                                            mag = {}
                                            jd = {}
                                            merr = {}
                                            snr={}
                                            
                                            output  = 'SURVEY:  LSST \n'                
                                            output += 'SNID: {}_{} \n'.format(zz[0],k)
                                            output += 'IAUC:    UNKNOWN \n'             
                                            output +=  'RA:     '+str(fieldRA)+'  deg \n'
                                            output +=  'DECL:   '+str(fieldDec)+'  deg \n'
                                            output +=  'MWEBV:    0.0  MW E(B-V) \n'
                                    #if psndata['zhost'] >0: 
                                            output +=  'REDSHIFT_FINAL:  '+str(z)+' +- '+'%5.3f' % self.z[2]+' (CMB)\n'
                                    #else:
                                    #    output +=  'REDSHIFT_FINAL:  0.0 +- 1.0 \n'
                    #output +=  'SEARCH_PEAKMJD:  {:.3f} \n'.format(mjd[0])
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
                                                detect= np.array(lc[ty][sty][sn][time]['detect'][filtermatch])
                                                mag[f] = lcMags[filtermatch][detect]
                                                jd[f] = obs[indexlc][filtermatch][detect]
                                                snr[f] = lcSNR[filtermatch][detect]
                                                merr[f] = 2.5*np.log10(1+1/snr[f])
                                                for h,j in enumerate(jd[f]):
                #            if merr[f][i]>0: 
                                                    fl = 10**(-0.4*(mag[f][h]))*1e11
                                                    if snr[f][h]>1: 
                #                flerr = 0.4*merr[f][i]*fl*log(10)
                                                        flerr = fl/snr[f][h]/1.3
                                                    else: 
                #                flerr = 10**(-0.4*(mag[f][i]))*1e11
                                                        flerr = fl/1.1
                                                    output += 'OBS: %9.3f   %s NULL  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f \n' % (j,f,fl,flerr,snr[f][h],mag[f][h],merr[f][h])
                                            output +='END: '
                                        dt = []
                                        dm = np.concatenate(np.array(dm))
                                        for f1 in np.unique(dataSlice[self.filterCol][indexlc]):
                                            filter2 = np.delete(np.unique(dataSlice[self.filterCol][indexlc]),np.where(np.unique(dataSlice[self.filterCol][indexlc])==f1)[0])
                                            fil1 = np.where(obs_filter[indexlc]==f1)[0]
                                            filtermatch1 = np.where(obs_filter[indexlc] == f1)[0]
                                            t1 = obs[indexlc][fil1][lcpoints_AboveThresh[filtermatch1]]
                                            for f2 in filter2: 
                                                fil2 = np.where(obs_filter[indexlc]==f2)[0]
                                                filtermatch2 = np.where(obs_filter[indexlc] == f2)[0]
                                                t2 = obs[indexlc][fil2][lcpoints_AboveThresh[filtermatch2]]
                                                dt.append([np.absolute(t1-x) for x in t2])
                                        dt= np.array([k for i in dt for k in i])  
                                        #print('lcAboveTresh={}'.format(len(np.where(lcpoints_AboveThresh==True)[0]))) 
                                        #print('obs_filter={}'.format(len(np.unique(obs_filter[indexlc][lcpoints_AboveThresh])))) 
                                        #print('dt_gap,{}'.format(len(len(np.where(dt<0.5)[0])))) 
                                        #print('dm_gap={}'.format(len(np.where(dm>0.3)[0])))
                                        #print('bin_t={}'.format(len(np.where(bin_t)[0])))
                                        if (len(np.where(lcpoints_AboveThresh==True)[0])>self.npoints) or (len(np.unique(obs_filter[indexlc][lcpoints_AboveThresh])) > self.nFilters): #or (len(np.where(dt<0.5)[0])>0) or (len(np.where(dm>0.3)[0])>0) or (len(np.where(bin_t)[0])>0):
                                            listout.write('LSST_{}_{}_{}.dat'.format(sn,zz[1],k)+'\n')
                                            ofile = open(os.path.join('./LC','LSST_{}_{}_{}.dat'.format(sn,zz[1],k)),'w')
                                            ofile.write(output)
                                            ofile.close()
                                            sn_list +=1
                                            nDetected += 1
                                        else:
                                            nNoDetected += 1
                
                
                                else:
                                    expl_t = np.delete(expl_t, np.where(expl_t==time))
                                    
                    nNoclass=0
                    if self.classify_zrange:
                        listout.close()
                        if sn_list >0:
                            filesnana=open(os.path.join('./SNANA_OUT','snana_classification_{}.dat'.format(zz[1])),'w')
                            p=subprocess.call([os.environ['SNANA_DIR']+'/bin/psnid.exe', os.environ['LSST_DIR']+'/PSNID_LSST.nml'], stdout=filesnana)
                            P=[]
                    #for ty in sngroups:
                            if os.path.exists('./SNANA_OUT/snana_classification_{}.dat'.format(zz[1])):     
                                f=open('./SNANA_OUT/snana_classification_{}.dat'.format(zz[1]),'r')
                                r=f.readlines()
                                for riga in r:
                                    loc= riga.split()
                                    if 'PBayes' in loc:
                                        if loc[1]=='Ia':
                                            pIa = float(loc[31]) 
                                        elif loc[1]=='Ibc':
                                            pIbc = float(loc[31])
                                        elif loc[1]=='II':
                                            pII = float(loc[31])
                                    if 'type' in loc:
                                        if loc[4]=='Ia':
                                            classify[z][0]+=1  # Ia
                                        elif loc[4]== 'Ibc':
                                            classify[z][1]+=1  # Ibc
                                        elif loc[4]=='II':
                                            classify[z][2]+=1  # II
                                        elif loc[4]=='UNKNOWN': 
                                            p = np.array([pIa,pIbc,pII])
                                            t =np.array(['Ia','Ibc','II'])
                                            if (np.sum(p)>0) and (np.sum(p)<1.1):
                                                if t[np.where(p==np.amax(p))]=='Ia':
                                                    classify[z][0]+=1  # Ia
                                                elif t[np.where(p==np.amax(p))]== 'Ibc':
                                                    classify[z][1]+=1  # Ibc
                                                elif t[np.where(p==np.amax(p))]=='II':
                                                    classify[z][2]+=1  # II
                                            else:
                                                nNoclass +=1
    
                            print('at z ={} for ty= {}: Ia={},Ibc={},II={}'.format(z, ty, classify[z][0],classify[z][1],classify[z][2]))
                            classify.update({'nlc':{z:[nDetected, nNoDetected]}})
                            classify.update({'noclass':{z:nNoclass}})
                            print([nDetected, nNoDetected,nNoclass])
                    
                          
        if self.dataout:
            classify.update({'explosion_time': np.array(expldist)})
            return classify

        else:
            N=np.sum([np.sum(classify[z]) for z in self.zrange])            
            return float(N)


#### this function plots the class object transientsmetric with the relative template LC
#### label is a list which each element is composed by a string with the sn template name
#### and the redshift at wich the LC is simulated.(e.g. '1990N z=0.1')
def singleplot(label, bDictlc, metriclcList,dbRuns):
        db = input('which database do you want to check?')
        label_split = pd.DataFrame([l.split() for l in label])
        template = input('which template do you want to plot? ')
        z = input('set z (e.g z=0.2): ')
        index = np.where((label_split[0]==template) & (label_split[1]==z))
        #loc =bundleDicts[db][(int(index[0][0]),'transientsmetric')].metricValues[np.where(bundleDicts[db][(int(index[0][0]),\
        #                                            'transientsmetric')].metricValues.mask == False)]
        bundle = np.array(bundleDicts[db].keys())[index]
        loc =pd.DataFrame(bundleDicts[db][bundle].metricValues[np.where(bundleDicts[db][bundle].metricValues.mask == False)])[0]
        pointing = input('set the index of the pointing (min={}, max={}):\
        '.format(min(range(len(loc))), max(range(len(loc)))))
        #indexpos = input('set the index of the position (min={}, max={}):\
        #'.format(min(range(len(loc[int(pointing)]['position']))),max(range(len(loc[int(pointing)]['position'])))))
        #position = tuple(loc[int(pointing)]['position'][int(indexpos)])
        dataloc = loc[int(pointing)]['lc']
        indextime = int(input('at which time (min index time: {}, max index time: {})?\
        '.format(min(range(len(dataloc.keys()))), max(range(len(dataloc.keys()))))))
        time = list(dataloc.keys())[int(indextime)]
        filterNames = [ 'g', 'r', 'i']
        colors = {'u': 'k', 'g': 'cyan', 'r': 'g', 'i': 'r', 'z': 'y', 'y': 'orange'}
        epoch = np.array(dataloc[time]['Epoch'])
        mjd = np.array(loc[int(pointing)]['expMJD'])
        mags = np.array(dataloc[time]['Mags'])
        msnr = np.array(dataloc[time]['SNR'])
        magerr=2.5*np.log10(1+1/msnr)
        filters = np.array(dataloc[time]['filter'])
        lc={}
        times=np.arange(-20,metriclcList[index[0][0]].transDuration,0.5)
        plt.figure(figsize=(10,9))
        for f in filterNames:
            lc[f]=metriclcList[index[0][0]].make_lightCurve(times,np.array([f]*len(times)))
            filtermatch = np.where(filters == f)
            detect= np.array(dataloc[time]['detect'][filtermatch])
            indT=np.where(detect==True)
            indF=np.where(detect==False)
            plt.plot(times+mjd-epoch[0], lc[f], color=colors[f])
            plt.scatter(x=epoch[filtermatch][indF]+mjd-epoch[0],y= mags[filtermatch][indF], marker='x',color=colors[f], label=f+' non-detect')
            plt.errorbar(x=epoch[filtermatch][indT]+mjd-epoch[0],y= mags[filtermatch][indT],yerr=magerr[filtermatch][indT], fmt='o',color=colors[f], label=f+' detect')
            plt.scatter(x=epoch[filtermatch][indT]+mjd-epoch[0],y= mags[filtermatch][indT],s=2, marker='o' ,color=colors[f])
            #plt.ylim(plt.ylim()[::-1])
        plt.ylim([27.,15.])
        plt.xlim([times[0]+mjd-epoch[0]-1,times[-1]+mjd-epoch[0]])
        #plt.xlim( times[0]  ,2+ times[0] +metriclcList[int(np.unique(index)[0])].transDuration+2 )
        plt.legend(ncol = 2, loc = (.7,.9), numpoints=1, fontsize='smaller') 
        plt.xlabel('MJD')
        plt.ylabel('Mags')
        plt.grid(b=None)
        plt.title('RA= {:.2f} DEC={:.2f} {} epoch0={:.2f}'.format(dataloc[time]['position'][0],dataloc[time]['position'][1], z, mjd-epoch[0])) 
        sv = input('Do you want to save the fig?')
        if sv in ['Y','YES','SI','y','si']:
            savename = input('input name of the file: ')
            plt.savefig(os.path.join('./Figure/'+savename+'.png'))
        return plt.show()
    
def skyplot(bDictlc,label, marker='.', xlabel='$\mathrm{RA[\degree]}$',ylabel='$\mathrm{DEC[\degree]}$',title='Pointing location'):
    RA,DEC=[],[]
    print(dbRuns)
    db = input('which database do you want to check?')
    for bundle in bundleDicts[db].keys():
        loc = pd.DataFrame(bundleDicts[db][bundle].metricValues[np.where(bundleDicts[db][bundle].metricValues.mask == False)])[0]
    
        for i ,data  in enumerate(loc):
            dataloc=data['lc']
            for time in dataloc.keys():
                RA.append(dataloc[time]['position'][0])
                DEC.append(dataloc[time]['position'][1])
    cmap = cm.rainbow
    plt.figure(figsize=(8,10))
    ax = plt.axes(projection='mollweide')
    #plt.scatter(np.degrees(360*np.array(RA_point)/2/np.pi-180),np.degrees(360*np.array(DEC_point)/2/np.pi), marker = '.',alpha=0.3, vmin = 0, vmax = 1.5, linewidths = 0., zorder = 10)
    plt.scatter(np.radians(360*np.array(RA)/2/np.pi-180),np.radians(360*np.array(DEC)/2/np.pi), marker = marker, vmin = 0, vmax = 1.5, linewidths = 0., zorder = 10)
    ax.set_title(title)
    ax.set_xlabel(r'{}'.format(xlabel))
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel(r'{}'.format(ylabel))
    ax.yaxis.label.set_fontsize(12)
    tick_labels = np.array([150, 120, 90, 60, 30, 0, -30, -90, -120, -150])
    tick_labels_y = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
    ax.set_xticklabels(tick_labels, zorder = 15)
    ax.set_yticklabels(tick_labels_y, zorder = 15)
    save= input('save fig? ')
    if save=='yes':
        name=input('name: ')
        plt.savefig('./Figure/'+name+'.png')
    return plt.show()

def hist_explosion(bundleDicts,bins=10, width=1, color='k'):
    print(dbRuns)
    db = input('which database do you want to check?')
    for bundle in bundleDicts[db].keys():
        loc = pd.DataFrame(bundleDicts[db][bundle].metricValues[np.where(bundleDicts[db][bundle].metricValues.mask == False)])[0]
        time = []
        und_time = []
        for i, data in enumerate(loc):
            dataloc=data['lc']    
            for t in dataloc.keys():
                    time.append(float(t)- data['expMJD'])
            for t in data['explDist']:
                und_time.append(float(t)-data['expMJD'])
    time = np.array(time)/365
    und_time = np.array(und_time)/365
    n=[]
    m=[]
    for j in np.arange(0,bins,width):
        o=0
        p=0
        for i in time:
            if i<=j+width and i>j:
                o+=1
            else:
                o+=0
        n.append(o)
        for i in und_time:
            if i<=j+width and i>j:
                p+=1
            else:
                p+=0
        m.append(p)
    
    plt.figure(figsize=(10,9))
    plt.bar(np.arange(0,bins,width),np.array(m)/np.sum(n), width=width,align='edge',alpha=0.5 ,linewidth=3,facecolor='darkorange', edgecolor='red', label='simulated explosion time')
    plt.bar(np.arange(0,bins,width),np.array(n)/np.sum(n),align='edge',width=width,alpha=0.5 , linewidth=3,facecolor='grey', edgecolor='black', label='observed explosion time')
    plt.legend()
    plt.grid(b=None)
    #plt.ylim([0,4100])
    #plt.axvline(float(org[0][0]), color='k', linestyle='dashed', linewidth=2)
    #plt.axvline(time[0], color=color, linestyle='dashed', linewidth=2)
    plt.xticks(np.arange(0,bins+width,width))
    plt.ylabel(r'$n_{SN}$')
    plt.xlabel('survey time (yrs)')
    save = input('save fig?')
    if save=='yes':
        name = input('name: ')
        plt.savefig('./Figure/'+name+'.png')
    return plt.show()
    
def pix2radec(nside, ipix, degree=True):
    """from ipix to ra/dec"""
    
    theta, phi = hp.pix2ang(nside, ipix)
    ra = phi
    dec = np.pi/2. - theta
    if degree:
        ra = np.rad2deg(ra)
        dec = np.rad2deg(dec)
    return ra, dec    

def plt_mlw(ra,dec,org=0,projection='mollweide',c= 'null', vmin=0,vmax=1, dicts={}):
    args={'colormap': 'rainbow','marker':'o','linewidths':0,'title':'mlw_plot','xlabel':'x','ylabel':'y', 'fontsize':12}
    if dicts:
        args.update(dicts)
        
    cmap = cm.get_cmap(args['colormap'])
    if c =='null':
        c= np.ones(len(ra))
    else:
        c=c
     
    x = np.remainder(ra+360-org,360) # shift RA values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=projection)
    ax.set_title(args['title'])
    ax.set_xticklabels(tick_labels)     # we add the scale on the x axis    
    ax.title.set_fontsize(12)
    ax.set_xlabel("RA")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Dec")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)
    im=plt.scatter(np.radians(x),np.radians(dec),c=c,cmap=cmap, marker = args['marker'], vmin = vmin, vmax = vmax, linewidths = args['linewidths'], zorder = 10)
    ax.tick_params('both', labelsize=14)
    cb=plt.colorbar(im,aspect=50,orientation='horizontal')
    for t in cb.ax.get_xticklabels():
         t.set_fontsize(12)