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
        
        
    def save_to_file(self, dic, filename="test_pkl.pkl"):
        '''save dict item to pickle file'''
        
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
        #check if all the filters for the observed lightcurves are available
        if all(np.in1d(self.observedFilter, dataSlice[self.filterCol])):
            index_filter = np.in1d( dataSlice[self.filterCol], self.observedFilter)
            obs_filter = dataSlice[self.filterCol][index_filter]
            obs = dataSlice[self.mjdCol][index_filter]        
            obs_m5 = m5[index_filter]
            expldist=[]
            nDetected = 0
            nNoDetected = 0
            nTransMax = 0
            jj=0
            for ty in sngroups:
                lc[ty]={}       
                if (self.classify_ty):    
                    listout=open(os.path.join('./LC','LSST.LIST'),'w')
                for sty in sngroups[ty]:
                    lc[ty][sty]={}
                    for sn in sngroups[ty][sty][0]:
                        lc[ty][sty][sn]={}
                        sn_list = 0
                        for j, z in enumerate(self.zrange):
                            if (self.classify_zrange):    
                                listout=open(os.path.join('./LC','LSST.LIST'),'w')
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
                                        lc[ty][sty][sn][time]['detect'] = np.ones(len(dataSlice), dtype=bool)
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
                                            output += 'SNID: {}_{} \n'.format(zz[0],jj)
                                            output += 'IAUC:    UNKNOWN \n'             
                                            output +=  'RA:     '+str(RA)+'  deg \n'
                                            output +=  'DECL:   '+str(DEC)+'  deg \n'
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
                                            output += 'NOBS: {} \n'.format(len(lcMags[(np.where((obs_filter[indexlc] == 'r') | (obs_filter[indexlc]=='g') | (obs_filter[indexlc]=='i')))])) 
                                            output += 'NVAR: 8 \n'
                                            output += 'VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR   SNR    MAG     MAGERR \n'
                                            for f in filterNames:
                                                filtermatch = np.where(obs_filter[indexlc] == f)
                                                detect= np.array(lc[ty][sty][sn][time]['detect'][filtermatch])
                                                indT=np.where(detect==True)[0]
                                                mag[f] = lcMags[indT]
                                                jd[f] = obs[indexlc][indT]
                                                snr[f] = lcSNR[indT]
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
                                            listout.write('LSST_{}_{}_{}.dat'.format(zz[0],zz[1],jj)+'\n')
                                            ofile = open(os.path.join('./LC','LSST_{}_{}_{}.dat'.format(zz[0],zz[1],jj)),'w')
                                            ofile.write(output)
                                            ofile.close()
                                            sn_list +=1
                                            jj+=1
                                            nDetected += 1
                                        else:
                                            nNoDetected += 1
                
                
                                else:
                                    expl_t = np.delete(expl_t, np.where(expl_t==time))
                        
                        if self.classify_zrange:
                            listout.close()
                            if sn_list >0:
                                filesnana=open(os.path.join('./SNANA_OUT','snana_classification_{}_{}.dat'.format(zz[0],zz[1])),'w')
                                p=subprocess.call([os.environ['SNANA_DIR']+'/bin/psnid.exe', os.environ['LSST_DIR']+'/PSNID_LSST.nml'], stdout=filesnana)
                                P=[]
                            #for ty in sngroups:
                                if os.path.exists('./SNANA_OUT/snana_classification_{}_{}.dat'.format(zz[0],zz[1])):
                                    f= open('./SNANA_OUT/snana_classification_{}_{}.dat'.format(zz[0],zz[1]))
                                else:
                                    continue
                                for riga in f:
                                    loc= riga.split()
                                    if 'PBayes' in loc:
                                            P.append(loc[loc.index('PBayes')+2])
                                    if 'best' in loc:
                                            if float(P[0])>float(P[1]) and float(P[0])>float(P[2]):
                                                classify[z][0]+=1  # Ia
                                            elif float(P[1])>float(P[0]) and float(P[1])>float(P[2]):
                                                classify[z][1]+=1  # Ibc
                                            elif float(P[2])>float(P[1]) and float(P[2])>float(P[0]):
                                                classify[z][2]+=1  # II
                                            else: 
                                                nNoDetected +=1
                                            P=[]
                                print('at z ={} for ty= {}: Ia={},Ibc={},II={}'.format(z, ty, classify[z][0],classify[z][1],classify[z][2]))
                            
                                
            if self.classify_ty:
                listout.close()
                filesnana=open(os.path.join('./SNANA_OUT','snana_classification_{}.dat'.format(ty)),'w')
                p=subprocess.call([os.environ['SNANA_DIR']+'/bin/psnid.exe', os.environ['LSST_DIR']+'/PSNID_LSST.nml'], stdout=filesnana)
                Ia=0
                Ibc=0
                II=0
                classify= {}
                classOK= {'Ia':0,'II':0,'Ibc':0}
                    
                
                P=[]
                            #for ty in sngroups:
                for sty in sngroups[ty]:
                    for sn in sngroups[ty][sty][0]:
                        if os.path.exists('./SNANA_OUT/snana_classification_{}.dat'.format(ty)):
                                f= open('./SNANA_OUT/snana_classification_{}.dat'.format(ty))
                        else:
                                continue
                        for riga in f:
                                loc= riga.split()
                                if 'PBayes' in loc:
                                          P.append(loc[loc.index('PBayes')+2])
                                if 'best' in loc:
                                    if float(P[0])>float(P[1]) and float(P[0])>float(P[2]):
                                                         Ia+=1
                                    elif float(P[1])>float(P[0]) and float(P[1])>float(P[2]):
                                                         Ibc+=1
                                    elif float(P[2])>float(P[1]) and float(P[2])>float(P[0]):
                                                         II+=1
                                    P=[]
                if Ia+Ibc+II==0:
                    nNoDetected +=1
                else:
                    classify[ty]=[Ia/(Ia+Ibc+II),Ibc/(Ia+Ibc+II),II/(Ia+Ibc+II)]
                    print('Ia={},Ibc={},II={}'.format(classify[ty][0],classify[ty][1],classify[ty][2]))
                    
                if ty == 'Ia':
                         classOK[ty]= Ia/nDetected
                if ty == 'II':
                         classOK[ty]= II/nDetected
                if ty == 'Ibc':
                         classOK[ty]= Ibc/nDetected
                    
                          
        if self.dataout:
            #if self.label:
            if self.classify_zrange:
                return classify
            elif self.classify_ty:
                self.explDist = expldist
                dicts = {'classifications': classify, 
                         'no_class':nNoDetected/nTransMax,
                         'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec))}
                if self.filename!=None:
                    self.save_to_file(lc, filename=self.filename)
                return dicts
        else:
            if self.classify_zrange:
                N=np.sum([np.sum(classify[z]) for z in self.zrange])
            elif self.classify_ty:
                N = np.sum([classOK[ty] for ty in ['Ia','Ibc','II']])
            return float(N)