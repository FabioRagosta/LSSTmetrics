description='show kcorrection for SN type.....'
import os,subprocess
import argparse
from numpy import *
import pylab

#sys.path.append('/home/enrico/scripts/alice')
#import alice

filt_in = 'UBVRI'
filt_ou = 'gri'
zrange = [0,1,.05]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description,\
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("sn",help='sn name')
    parser.add_argument("-v", "--verbose",dest="verbose",action="store_true",\
           default=False,help='Enable task progress report')


    args = parser.parse_args()

def read_snana():

    pid =subprocess.Popen("stilts tpipe in=kcor_SUDARE.fits#3 ofmt=ascii "+\
          "cmd='keepcols \" Trest Redshift AVwarp K_Ug K_Bg K_Ur K_Br K_Vr K_Ui K_Bi K_Vi K_Ri K_Rr \" '",stdout=subprocess.PIPE,shell=True)   
    output,error = pid.communicate()

    _catalog = output.split("\n")
    trest,redshift,AVwarp,kksnana = [],[],[],{}
    for f in ['Ug','Bg','Ur','Br','Vr','Ui','Bi','Vi','Ri', 'Rr', 'Ii']:
        kksnana[f]=[]
    for i in range(63601,66357):
        trest.append(float(_catalog[i].split()[0]))
        redshift.append(float(_catalog[i].split()[1]))
        AVwarp.append(float(_catalog[i].split()[2]))
        for j,k in enumerate(['Ug','Bg','Ur','Br','Vr','Ui','Bi','Vi','Ri','Rr']):
            kksnana[k].append(float(_catalog[i].split()[j+3]))

    for k in kksnana.keys(): kksnana[k] = array(kksnana[k])

    return array(trest),array(redshift),array(AVwarp),kksnana


if __name__ == "__main__":

    sn = args.sn

    trest,redshift,AVwarp,kksnana= read_snana()

    kph,kz,kcor = {},{},{}
    ff = open('kk/'+sn+'.kk')
    righe = ff.readlines()
 
    igf = []
    for i,r in enumerate(righe):
        if '---' in r: igf.append(i-1)

    for n,i in enumerate(igf):
        gf = righe[i].split()[0]
        kz[gf],kcor[gf] = [],[]
        i1 = len(righe)
        if n<len(igf)-1: i1 = igf[n+1]
        kph[gf] = array([float(p) for p in righe[i+1].split()[1:]])

        for r in righe[i+2:i1]:
            kz[gf].append(float(r.split()[0]))
            kcor[gf].append(array([float(k) for k in r.split()[1:]]))
    ff.close()

    pylab.ion()
    for k in kz.keys():
        for i,z in enumerate(kz[k]):
            pylab.plot(kph[k],kcor[k][i],'r-')
        
            ii = where(abs(redshift-z)<.0001)
            pylab.plot(trest[ii],kksnana[k[::-1]][ii],'g-')

            pylab.title(k+'   '+str(z))
            raw_input('.... next ....')
            pylab.clf()
