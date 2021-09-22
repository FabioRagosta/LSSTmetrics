description='fit kcorrection for SN type.....'
import os,sys
import time
import argparse
from numpy import *
import pylab
import kcor_misu

#filt_in = 'gri'
filt_in = 'BVR'
zrange = [0,1,.05]

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description=description,\
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("sn",help='sn name')
    parser.add_argument("-g", "--filt_ou",dest="filt_ou",default='gri',
            type=str,help='filter out')

    parser.add_argument("-v", "--verbose",dest="verbose",action="store_true",\
           default=False,help='Enable task progress report')

    args = parser.parse_args()

def smooth(xx,yy,kphfit,wind):

    xfit = xx[argsort(xx)]
    _yy = yy[argsort(xx)]

    yfit = []
    for x in xfit:
        ii = []
        _wind = wind
        while size(ii)<5:
            ii = where(abs(xfit-x)<_wind)
            _wind *= 1.5
        yfit.append(mean(_yy[ii]))
    kcorfit = interp(kphfit,xfit,yfit)

    return kcorfit

class kcor_fit:

    def __init__(self):

        self.xx,self.yy = array([]),array([])
        for sn in kph.keys():
            if gf in kz[sn].keys():
                if _zz in kz[sn][gf]: 
                    jz = kz[sn][gf].index(_zz)
                    ii = where((kcor[sn][gf][jz]<100)&(kph[sn][gf]<120))
                    self.all = ax.plot(kph[sn][gf][ii],kcor[sn][gf][jz][ii],\
                                           'o',label=sn)        
                    self.xx = concatenate((self.xx,kph[sn][gf][ii]))
                    self.yy = concatenate((self.yy,kcor[sn][gf][jz][ii]))
        if len(self.xx)==0: 
            phfit,yfit = [],[]
            return
        ax.legend(numpoints=1,loc=(.9,.0))
        self.idd = self.xx>-100
        self.testo = pylab.figtext(0.3,.91,'filter='+gf+'    z='+str(_zz))
        self.selected, = ax.plot(self.xx[invert(self.idd)],\
                               self.yy[invert(self.idd)],'xr',ms=20)
        self.fitline, = ax.plot([],[],'-')

        if len(klastph):  ax.plot(klastph,klastcor,'--')
        self.fit()

    def onclick(self,event):
        if not event.inaxes: return
        xdata,ydata = event.xdata,event.ydata
        distx,disty = abs(xdata-self.xx),abs(ydata-self.yy)
        rx,ry = max(self.xx)-min(self.xx),max(self.yy)-min(self.yy)
        ii = argmin((distx/rx)**2+(disty/ry)**2)
        if distx[ii]>10 or disty[ii]>.1: return
        if event.button == 1:  self.idd[ii] = False
        if event.button == 2:  self.idd[ii] = True
        self.selected.set_data(self.xx[invert(self.idd)],self.yy[invert(self.idd)])
        self.fit()

    def onkeypress(self,event):
        global deg,meth
        if event.key in '0123456789':
            meth = 'poly'
            deg = int(event.key)
        if event.key == 'm': meth = 'smooth'
        self.fit()

    def fit(self):
        global deg,meth,phfit,yfit        
        phfit = arange(kphra[gf][0],kphra[gf][1]+1,5)
        if len(self.xx[self.idd])<8: meth='poly'

        if meth == 'smooth':
            wind = 3.
            yfit = smooth(self.xx[self.idd],self.yy[self.idd],phfit,wind)
            self.fitline.set_data(phfit,yfit)
            _yfit  = interp(self.xx[self.idd],phfit,yfit)
            rms = sqrt((sum(_yfit-self.yy[self.idd])**2)/float(len(self.yy[self.idd])))
        elif meth=='poly':
            if deg > len(self.xx[self.idd])-1: 
                deg = max(1,len(self.xx[self.idd])-1)
                print 'warning: deg=',deg
            pol = polyfit(self.xx[self.idd],self.yy[self.idd],deg)
            yfit= polyval(pol,phfit)
            self.fitline.set_data(phfit,yfit)
            rms = std(self.yy[self.idd]-polyval(pol,self.xx[self.idd]))

        pylab.setp(self.testo,text='filter='+gf+'    z='+str(_zz)+'    rms='+\
                       str(round(rms,2)))
        fig.canvas.draw()


if __name__ == "__main__":

    global deg,meth,phfit,yfit        

    sne = args.sn.split(',')
    filt_ou = args.filt_ou

    zed,kph,kz,kcor = {},{},{},{}  # read  kcorrections
    kphra = {}
    for sn in sne:
        kph[sn],kz[sn],kcor[sn] = {},{},{}
        ff = open('kk/'+sn+'.kcor')
        righe = ff.readlines()
        zed[sn]= float(righe[0].split()[1])

        igf = []
        for i,r in enumerate(righe):
            if '---' in r: igf.append(i-1)

        for n,i in enumerate(igf):
            gf = righe[i].split()[0]
            kz[sn][gf],kcor[sn][gf] = [],[]
            if gf not in kphra.keys(): kphra[gf] = [100,0]
            i1 = len(righe)
            if n<len(igf)-1: i1 = igf[n+1]
            kph[sn][gf] = array([float(p) for p in righe[i+1].split()[1:]])
            if min(kph[sn][gf]) < kphra[gf][0]: kphra[gf][0]=int(min(kph[sn][gf])) 
            if max(kph[sn][gf]) > kphra[gf][1]: kphra[gf][1]=int(max(kph[sn][gf])) 
            
            for r in righe[i+2:i1]:
                kz[sn][gf].append(float(r.split()[0]))
                kcor[sn][gf].append(array([float(k) for k in r.split()[1:]]))
        ff.close()

    pylab.ion()
    fig = pylab.figure()

    kzfit,kphfit,kcorfit = {},{},{}
    deg = 4
    for g in filt_ou:
        zra = kcor_misu.assign_filter(g,filt_in,zrange)
        for f in filt_in[::-1]:
            klastph,klastcor = '',''
            if f not in zra.keys(): continue

            meth = 'smooth'
            j=0
            while j < len(zra[f]): 
                ax = pylab.axes()
                for sn in kph.keys():
                    if g+f in kph[sn].keys(): 
                        if g+f not in kphfit.keys(): 
                            kzfit[g+f],kphfit[g+f],kcorfit[g+f] = [],{},{}

                if g+f not in kphfit.keys(): break
                if  kphra[g+f][1]>100: kphra[g+f][1]=100
                kzfit[g+f].append(zra[f][j])
                print '###',g+f,zra[f][j]
                if j>0: 
                    klastph,klastcor = kphfit[g+f][zra[f][j-1]],\
                        kcorfit[g+f][zra[f][j-1]]
                answ = 'r'
                while answ not in 'nq':   
                    gf = g+f
                    _zz = zra[f][j]
                    fitkk = kcor_fit()
                    fig.canvas.mpl_connect('button_press_event',fitkk.onclick)
                    fig.canvas.mpl_connect('key_press_event',fitkk.onkeypress)
                    
                    _answ = raw_input('p-reviuos or n-ext [n] ? ')
                    kphfit[g+f][zra[f][j]],kcorfit[g+f][zra[f][j]] = phfit,yfit
                    if _answ:
                        if _answ=='p': 
                            j += -1
                            answ='n'
                    else: 
                        answ='n'
                        j +=1
                    pylab.clf()

                
    ff = open('kfit.kk','w')
    for gf in kphfit.keys():
        ff.write(gf+' \n')
        ff.write(' ---  ')
        z0 = kphfit[gf].keys()[0]
        for j in range(len(kcorfit[gf][z0])):
            ff.write('%6.1f ' % kphfit[gf][z0][j])
        ff.write('\n')
        for z in kzfit[gf]:
            ff.write('%5.2f ' % z)
            for j in range(len(kcorfit[gf][z])):
                ff.write('%6.2f ' % kcorfit[gf][z][j])
            ff.write('\n')
    ff.close()

    print "********** Completed in ",int(time.time()-start_time),"sec"



