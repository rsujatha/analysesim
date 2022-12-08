from __future__ import division
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import integrate
from scipy import stats
import multiprocessing as mp

class cosmology(object):
	def __init__(self,Omega_matter = 0.276,Omega_lambda = 0.724,H_0=70.,ns=0.961,sigma_8 = 0.811 ,Omega_baryon = 0.045 ):
		self.Omega_matter=Omega_matter
		self.Omega_lambda=Omega_lambda
		self.H_0=H_0      ### in km/sec/Mpc
		self.hubble=self.H_0/100.
		self.rho_c_si = 9.5e-27    ## in SI units ?? why do i need this here
		self.rho_c_h2_msun_mpc3 = 2776*1e8    ## in (msun/h)(mpc/h)**3 is the critical density today (ie redshift evolution not included)
		self.ns =0.9677 if ns=='planck18' else ns
		self.sigma_8 = 0.815 if sigma_8=='planck18' else sigma_8
		self.Omega_baryon = Omega_baryon
		self.Mpc_to_m = 3.0857e22  ### multiply this to Mpc to convert it to meters
		self.sec_to_Gyr = 1/(365.25*60*60*24*10**9)   ### multiply this with sec to convert it to GigaYears
		self.speed_of_light = 299792.458   ### in kilometers per sec
		self.radiationenergyconstant = 7.56577e-16 ## Jm^(-3)K^(-4)
		self.delta_c=1.686
		#~ self.NPROC = 
	def H(self,a,ok=0):
		# ~ return self.H_0*(self.Omega_matter*a**(-3) + self.Omega_lambda + ok*(a)**(-2)+(1-self.Omega_lambda-self.Omega_matter)*a**(-4))**(1/2)
		if ok==0:
			# ~ return self.H_0*(self.Omega_matter*a**(-3) + (1-self.Omega_matter-9.23640e-5)+(9.23640e-5)*a**(-4))**(1/2)
			return self.H_0*(self.Omega_matter*a**(-3) + (1-self.Omega_matter-2.472/(0.7*0.7)*1e-5)+(2.472/(0.7*0.7)*1e-5)*a**(-4))**(1/2)
		else:
			return self.H_0*(self.Omega_matter*a**(-3) + (1-self.Omega_matter-ok-8e-5) + ok*(a)**(-2)+(8e-5)*a**(-4))**(1/2)
	
	def E(self,z,ok=0):
		"""
		seems like H/H_0 need to confirm
		"""
		a = 1./(1.+z)
		# ~ return self.H_0*(self.Omega_matter*a**(-3) + self.Omega_lambda + ok*(a)**(-2)+(1-self.Omega_lambda-self.Omega_matter)*a**(-4))**(1/2)
		if ok==0:
			# ~ return self.H_0*(self.Omega_matter*a**(-3) + (1-self.Omega_matter-9.23640e-5)+(9.23640e-5)*a**(-4))**(1/2)
			return (self.Omega_matter*a**(-3) + (self.Omega_lambda))**(1/2)
		else:
			return (self.Omega_matter*a**(-3) + (1-self.Omega_matter-ok-8e-5) + ok*(a)**(-2)+(8e-5)*a**(-4))**(1/2)


	def BryanNorman_del_crit(self,z):
		x = self.Omega_matter*(1+z)**3/(self.E(z))**2-1
		if self.Omega_lambda ==0:
			return 18*np.pi**2+60*x-32*x**2
		else:     ### in the paper this condition is given as or == 0, pls take care while using this condition
			return 18*np.pi**2+82*x-39*x**2


	def Omega_radiation(self,TCMB=2.7255):
		radiation_density = self.radiationenergyconstant*TCMB**4/(self.speed_of_light*1e3)**2


	def GrowthFunctionAnalytic(self,a):
		"""
		check why is H defined as a series on ones????
		"""
		a=np.array(a)+1e-15
		D=np.ones(np.size(a))
		H=np.ones(np.size(a))
		H=self.H_0*(self.Omega_matter*(a**(-3))+self.Omega_lambda)**(1/2.)
		D=(H/self.H_0)*a**(5/2.)/np.sqrt(self.Omega_matter)*sp.hyp2f1(5/6.,3/2.,11/6.,-a**3*self.Omega_lambda/self.Omega_matter)
		return D


	def f(self,a):
		## f = dln D/dln a      ----- where D is the Growth function
		f = -(self.H_0**2/cosmology.H(self,a)**2)*self.Omega_matter*(3/2.*a**(-3.) - 5/(2.*a**2*cosmology.GrowthFunctionAnalytic(self,a)))
		return f

####### BBKS TRANSFER FUNCTION: Dodelson Chapter: Inhomogeneities , Page 205  ######################
	def BBKS_tf(self,k):
		#Gamma = 	self.Omega_matter*self.hubble  #from Dodelson
		Gamma = 	self.Omega_matter*self.hubble * np.exp(-self.Omega_baryon-self.Omega_baryon/self.Omega_matter)  ## correction from sugiyama1994
		q = k/Gamma
		return np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(16.2*q)**2+(5.47*q)**3+(6.71*q)**4)**(-0.25)

	def Wk(self,k,R):
		"""
		Fourier Transform of a Spherical Top Hat Filter
		"""
		return 3/(k*R)**3*(np.sin(k*R)-(k*R)*np.cos(k*R))
		
	def wk(self,x):
		"""
		Fourier Transform of a Spherical Top Hat Filter
		"""
		return 3/(x)**3*(np.sin(x)-(x)*np.cos(x))
		
	def PS(self,k,z,T):
		"""
		Input
		k in h Mpc^1
		z redshift
		T the tranfer function
		
		Outputs 
		Pk the power spectrum
		"""
		R=8.
		integrand = 1/(2*np.pi**2)*k**(self.ns+2.)*T**2*cosmology.Wk(self,k,R)**2
		igrate = np.trapz(integrand,k)
		SigmaSquare=self.sigma_8**2
		NormConst = SigmaSquare/igrate
		return NormConst*k**self.ns*(T)**2*cosmology.GrowthFunctionAnalytic(self,1./(1.+z))**2/cosmology.GrowthFunctionAnalytic(self,1)**2


	def GaussianWk(self,k,R):
		### A normalised gaussian 1/sqrt(2pi)R exp(-k^2R^2/2) 
		### R is the standard deviation of the Gaussian.
		"""
		This function takes standard deviation of a normalised gaussian as input and gives its fourier transform as output
		"""
		return np.exp(-k**2*R**2/2)


	def PS_calc(self,BoxSize,NofParticles,deltax,NofBins=100,kmin=None,kmax=None,cic_flag=1):
		"""
		Calculates power spectrum P(k) 
		Input: 
		BoxSize - Size of the box 
		deltax  - real space density fluctuation 
		NofBins - number of k bins
		cic_flag - if the deltax is computed using cic then window corrections are done to the power spectrum
		
		Output:
		Pk   -  P(k)  
		k_array -  k in h Mpc^{-1}
		"""
		
		dk = 2*np.pi/BoxSize
		GridSize = deltax.shape[0]
		kspace = np.concatenate([range(0,int(GridSize/2)),range(-int(GridSize/2),0)])*dk	
		
		if kmin==None:
			kmin = dk
		if kmax==None:
			kmax = GridSize/2*dk
		
		k_bin = np.logspace(np.log10(kmin),np.log10(kmax),NofBins+1)
		#print (k_bin)
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace[0:int(GridSize/2)+1], indexing='ij',sparse=True)
		k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
		deltak = np.fft.rfftn(deltax)/(GridSize)**3   ### deltak_{our convention} = 1/GridSize**3  deltak_{numpy}(See Aseem's Math Methods Notes and numpy fft documentation)
		### correcting for cic density field 
		if cic_flag==1:
			deltak /= (np.sinc(BoxSize*k_x/(2*np.pi*GridSize))*np.sinc(BoxSize*k_y/(2*np.pi*GridSize))*np.sinc(BoxSize*k_z/(2*np.pi*GridSize)))**2
		
		Pk = deltak*deltak.conjugate()*BoxSize**3 ## P(k) = BoxSize**3 |deltak_{our convention}|**2
		

		Pk = Pk.flatten()
		k = k.flatten()
		### The following line will bin the P(k) logarithmically and average over values belonging to the same bin
		Pk_avg = np.histogram(k , bins = k_bin , weights = Pk)[0]/ (np.histogram(k , bins = k_bin )[0]+1e-15)
		
		#### correcting for shot noise
		Pk_avg -= BoxSize**3/(NofParticles)
		
		
		return Pk_avg, np.sqrt(k_bin[:-1]*k_bin[1:])
		
	def T10(self,argument,k,Tfn,z=0,mass_flag=1):
		"""
		Input:
		k in h Mpc^1
		T the tranfer function
		argument can either be \nu or mass in Mpc/h
		Output:
		returns Tinker2010 bias
		mass_flag 1 takes mass as input
		"""	
		delta_c = 1.686
	
		if mass_flag ==1:
			R = (3/(4.*np.pi)*argument/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.)
			PS = cosmology.PS(self,k,z,Tfn)
			Delk = 1/(2.*np.pi**2)*PS*k**3.
			sigma_square = np.zeros([len(R),1])
			for i in range(0,len(R)):
				wk = cosmology.Wk(self,k,R[i])
				sigma_square[i] = np.trapz(Delk*wk**2/k,k)
			v = delta_c/np.sqrt(sigma_square)
		else:
			v = argument
		delta = 200
		y = np.log10(delta)
		A = 1. + 0.24 * y * np.exp(-(4/y)**4)
		a = 0.44 * y - 0.88
		B = 0.183
		b = 1.5
		C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4/y)**4)
		c = 2.4
		bias = 1- A*v**a/(v**a+delta_c**a)+B*v**b+C*v**c
		return bias.flatten()
	
	def DiemerKravtsov15(self,argument,k,Tfn,z=0,get='median',mass_flag=1):
		"""
		Refer to https://iopscience.iop.org/article/10.1088/0004-637X/799/1/108
		Input:
		argument: mass in Mpch^-1
		"""
		delta_c = 1.686
		kappa = 0.69
		if get =='median':
			phi0 = 6.58
			phi1 = 1.37
			eta0 = 6.82
			eta1 = 1.42
			negativealpha = -1.12
			beta = 1.69
		if mass_flag ==1:
			R = (3/(4.*np.pi)*argument/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.) ##is in Mpch^-1 
			PS = cosmology.PS(self,k,z,Tfn)
			Delk = 1/(2.*np.pi**2)*PS*k**3.
			sigma_square = np.zeros([len(R),1])
			for i in range(0,len(R)):
				wk = cosmology.Wk(self,k,R[i])
				sigma_square[i] = np.trapz(Delk*wk**2/k,k)
			v = delta_c/np.sqrt(sigma_square)
			kR = kappa*2*np.pi/R     ##is in Mpc^-1 h
			dlnP_by_dlnk = np.diff(np.log(PS))/np.diff(np.log(k))
			index =  np.searchsorted(k,kR)    ### "index" is an array of size(kR), it gives the index of the k array where the value is closest and greater than each kR
			# ~ n =  np.sign(dlnP_by_dlnk[index])*np.sqrt(dlnP_by_dlnk[index]*dlnP_by_dlnk[index-1])
			n =  (dlnP_by_dlnk[index]+dlnP_by_dlnk[index-1])/2
		cmin = phi0 + phi1*n
		vmin = eta0 + eta1*n
		c200c = cmin/2*((v.flatten()/vmin)**negativealpha+(v.flatten()/vmin)**beta)	
		print (c200c)
		return c200c
	
	def PeakHeight(self,mass,k,Tfn,z):
		"""
		Inputs:
		mass in Msunh-1
		k in h Mpc^1
		z redshift
		T the tranfer function
		"""
		R = (3/(4*np.pi)*mass/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.) ## is in units of Mpch-1
		PS = cosmology.PS(self,k,z,Tfn)
		sigma_square = np.zeros([len(R),1])
		for i in range(0,len(R)):
			wk = cosmology.Wk(self,k,R[i])
			sigma_square[i] = 1/(2.*np.pi**2)*np.trapz(PS*wk**2*k**2,k)
		nu = self.delta_c/np.sqrt(sigma_square)
		return nu.flatten()
			
	def dWk2_by_dR(self,k,R):
		return -54/(k**6.*R**7.)*(np.sin(k*R)-(k*R)*np.cos(k*R))**2. + 18/(k**4.*R**5.) * (np.sin(k*R)-(k*R)*np.cos(k*R)) * np.sin(k*R)

	def T08(self,mass,k,Tfn,z):
		"""
		Refer to Tinker (2008): https://arxiv.org/pdf/0803.2706.pdf
		mass should be in Msun h^-1
		"""
		R = (3/(4*np.pi)*mass/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.)
		PS = cosmology.PS(self,k,z,Tfn)
		Delk = 1/(2*np.pi**2)*PS*k**3
		sigma_square = np.zeros([np.size(R),1])		
		rho_dln_simgainv_by_dm = np.zeros([np.size(R),1])	
		
		for i in range(0,np.size(R)):
			wk = cosmology.Wk(self,k,R[i])
			dWk2_by_dR = cosmology.dWk2_by_dR(self,k,R[i])
			sigma_square[i] = 1/(2.*np.pi**2)*np.trapz(PS*wk**2*k**2,k)
			rho_dln_simgainv_by_dm[i] = - 1/(2*sigma_square[i]) * np.trapz(Delk/k*dWk2_by_dR/(4*np.pi*R[i]**2),k)
			
		sigma = np.sqrt(sigma_square)
		delt = 200
		A = 0.186*(1+z)**(-0.14)
		a = 1.47*(1+z)**(-0.06)
		alpha = 10**(-(0.75/(np.log10(delt/75)))**(1.2))
		b = 2.57*(1+z)**(-alpha)
		c = 1.19
		f = A* ((sigma/b)**(-a)+1)*np.exp(-c/sigma**2)
		dn_by_dlnm = f * rho_dln_simgainv_by_dm
		return dn_by_dlnm


	def massfuncbiasTinker(self,mtab,Dlin,ktab,dlnk,Om=0.25,z=0.0,Delta=200.0):
		""" Tinker++ mass function and linear Eulerian bias
			at redshift z, evaluated on given mass grid.
			Returns dn/dlnm and bias.
			Parameters calibrated for mMean200
		"""
		dcsph = 1.686
		dc0 = dcsph
		# Should be 1.686.
		a = 1/(z+1.0)
		dc = dc0*cosmology.GrowthFunctionAnalytic(self,a=1)/cosmology.GrowthFunctionAnalytic(self,a=a)
		
		# # valid for Delta=200
		# aA0 = 0.186
		# ab0 = 2.57
		# aa0 = 1.47
		# ac = 1.19
		# bA = 1.0000597
		# ba = 0.13245
		# bB = 0.183
		# bb = 1.5
		# bC = 0.26523
		# bc = 2.4
		
		lgDelta = np.log10(Delta)
		aA0 = (0.1*lgDelta-0.05) if Delta < 1600.0 else 0.26
		aa0 = 1.43 + (lgDelta-2.3)**1.5
		ab0 = 1.0 + (lgDelta-1.6)**(-1.5)
		ac = (1.2 + (lgDelta-2.35)**1.6) if lgDelta > 2.35 else 1.19
		alpha = 10**(-(0.75/np.log10(Delta/75.))**1.2)
		
		
		aA = aA0*(1+z)**(-0.14)
		aa = aa0*(1+z)**(-0.06)
		ab = ab0*(1+z)**(-alpha)
		
		qTink = 2*ac/dc0**2
		
		expfac = np.exp(-(4/lgDelta)**4)
		bA = 1.0 + 0.24*lgDelta*expfac
		ba = 0.44*lgDelta - 0.88
		bB = 0.183
		bb = 1.5
		bC = 0.019 + 0.107*lgDelta + 0.19*expfac
		bc = 2.4
		
		Rtab = (mtab/(4*np.pi*Om*self.rho_c_h2_msun_mpc3/3.0))**(1/3.0)
		# Lagrangian R-values in Mpc/h corresponding to bin centers
		kR = np.outer(Rtab,ktab)
		W = cosmology.wk(self,kR)
		Wsq = W**2
		sig02 = np.trapz(Wsq*Dlin, dx=dlnk, axis=1)
		mdWdR = -Wthpr(kR)*ktab
		sig12 = ny.trapz(W*mdWdR*Dlin, dx=dlnk, axis=1)/Rtab
		jacob = Rtab**2*sig12/sig02/3.0
		# |dlns/dlnm|/2 = jacob
		#               = 1/6 |dlns/dlnR|
		#               = -1/6 R/s ds/dR
		
		nu2 = qTink*dc**2/sig02
		# List of nu^2 values multiplied by qTink
		sigtabz = ny.sqrt(sig02)*(dc0/dc)
		vfv = aA*(1+(sigtabz/ab)**(-aa))*ny.exp(-0.5*nu2)
		
		mfTink = vfv*jacob*Om*rhoc/mtab
		# dn/dlnm
		nu = dc0/sigtabz
		bTink = 1-bA/(1+sigtabz**ba) + bB*nu**bb + bC*nu**bc
		# Linear Eulerian bias
		
		return mfTink,bTink
	
	
	def weighted_bT10(self,bins,massarray,k,Tfn,z):
		"""
			mass should be in Msun h^-1 and bins log spaced
		"""
		bavg = np.zeros([np.size(bins)-1])
		for i in range(np.size(bins)-1):
			subbin = np.logspace(np.log10(bins[i]),np.log10(bins[i+1]),100)
			b1     = cosmology.T10(self,subbin,k,Tfn,z).flatten() 
			dndlnm = cosmology.T08(self,subbin,k,Tfn,z).flatten()
			dlnm   = np.mean(np.log(subbin[1:]/subbin[0:-1]))
			bavg[i] =  np.trapz(b1*dndlnm,dx = dlnm)/np.trapz(dndlnm,dx = dlnm)
		mdn = stats.binned_statistic(massarray,massarray, statistic='median', bins=bins, range=None)[0]	
		return bavg,mdn
		
	def weighted_bT08(self,bins,massarray,k,Tfn):
		"""
		Note:
		This code assumes that the mass array is logarithmically spaced
		
		Input:
		massarray in Msun h^-1
		Volume in Mpc^3 h^-3
		"""
		dndlnm_avg = np.zeros([np.size(bins)-1])
		bins = np.array(bins)
		Dlnm=np.mean(np.log(bins[1:]/bins[0:-1]))
		for i in range(np.size(bins)-1):
			subbin = np.logspace(np.log10(bins[i]),np.log10(bins[i+1]),100)
			dndlnm = cosmology.T08(self,subbin,k,Tfn).flatten()
			dlnm   = np.mean(np.log(subbin[1:]/subbin[0:-1]))
			dndlnm_avg[i] =  np.trapz(dndlnm,dx = dlnm)/(Dlnm)
		mdn = stats.binned_statistic(massarray,massarray, statistic='median', bins=bins, range=None)[0]	
		return dndlnm_avg,mdn


	def b2(self,b1):
		"""
		m equation 5.2 of Lazeyras et al. (2016b, L16)
		"""
		return 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3
			
	def R200b(self,M200b):
		"""
		Takes M200b in Msun h^-1 and returns R200b in kpc/h
		Note!! R200b is in comoving coordinates, and it has no redshift evolution!!
		"""
		return (M200b/(4./3.*np.pi*self.rho_c_h2_msun_mpc3*self.Omega_matter*200))**(1./3.)*1000.  ### multiplying by thousand here to keep it consistently in kpc h^-1
	
	def R200c(self,M200c,z=0):
		"""
		Takes M200c in Msun h^-1 and returns R200c in kpc/h
		Note!! R200c is in comoving coordinates. Hence it has (1+z) evolution in addition to E(z)^2
		"""
		return (M200c/(4./3.*np.pi*self.rho_c_h2_msun_mpc3*cosmology.E(self,z)**2*200))**(1./3.)*1000.*(1.+z)  ### multiplying by thousand here to keep it consistently in kpc h^-1

	def M200b(self,R200b):	
		"""
		Takes R200b in Mpc h^-1 and returns M200b in Msun/h
		"""	
		return (4./3.*np.pi*self.rho_c_h2_msun_mpc3*self.Omega_matter*200)*R200b**3	
		
	def dt(self,z):
		"""
		dt = da*dt/da = da * dt/da *a *1/a = da/H *1/a = -dz/H *1/(1+z)  
		"""
		return - 1.0/((1+z)*cosmology.H(self,1/(1+z)))	
		
	
	def CosmicTime(self,zintrest):
		"""
		gives age of the universe at the redhift of intrest
		should integrate z from infinity to zofintrest
		"""
		time_in_sec =  integrate.quad(lambda z: cosmology.dt(self,z), np.inf, zintrest)[0] * self.Mpc_to_m * 1e-3
		return time_in_sec * self.sec_to_Gyr
		
	def LookbackTime(self,zintrest):
		"""
		gives the time in Gyr calculated from the redshift of intrest (in the past) to the present time.
		"""
		time_in_sec = integrate.quad(lambda z: cosmology.dt(self,z), zintrest, 0)[0] * self.Mpc_to_m * 1e-3
		return time_in_sec * self.sec_to_Gyr		
		
	def ComovingDistance(self,zintrest,omegak = 0):
		"""
		for light the interval ds^2 =0, and imagine it travelling along the r coordinate and not changing in phi and theta.
		0 = -dt^2 + a^2 dr^2/(1-kr^2) + 0
		solving for dr gives the comoving distance in Mpc
		"""	
		chi = integrate.quad(lambda z: self.speed_of_light/cosmology.H(self,1.0/(1.0+z),omegak), 0, zintrest)[0]   ## Chi in Mpc	
		k = -omegak * self.H_0**2/self.speed_of_light**2
		
		if k ==0:
			k = 1e-16
			def S_k(x):
				return x
		elif k<0:
			def S_k(x):
				return np.sinh(x)
		elif k>0:
			def S_k(x):
				return np.sin(x)
		return 1/np.sqrt(np.abs(k))*S_k(np.sqrt(np.abs(k))*chi)

		
	def ConformalDistance(self,zintrest,zobs):	
		return integrate.quad(lambda z: self.speed_of_light/cosmology.H(self,1.0/(1.0+z)), zobs, zintrest)[0] ## Chi in Mpc
	
	def ConformalDistance_EdS(self,z,zobs):
		return 2*self.speed_of_light/self.H_0*(1/(1+zobs)**(1/2.)-1/(1+z)**(1/2.))
		
	def ParticlesHorizon(self,z):	
		return cosmology.ConformalDistance(self,np.inf,z)

#### to do : make contour plot oof om vs ok for diff ol values of particle horizon at z=0		
		
	
		
	def LuminosityDistance(self,zintrest,omegak=0):
		"""
		returns the luminosity distance D_l which is the distance measured by observing the magnitude of standard candles in the sky 
		D_l = F/4piL F-> flux observed , L -> luminosity expected from nearby objects
		D_l = (1+z)Chi  Chi-> comoving distance from the object
		"""
		return (1+zintrest)*cosmology.ComovingDistance(self,zintrest,omegak) 		
	
	def AngularDiameterDistance(self,zintrest,omegak=0):
		return cosmology.ComovingDistance(self,zintrest,omegak)/(1+zintrest)
	
		
	def pool_it_1d(self,func_str,zipped_args,arg_size):
		""" Convenience function to evaluate generic function of 1-d array arguments using Pool.
				func_str: function name as string
				zipped_args: zip(arg1,arg2,...) where arg1,arg2,... are 1-d arrays of equal length
				arg_size: common length of argument arrays.
				Returns evaluation of function as array of size arg_size.
		"""
		out = np.zeros(arg_size,dtype=float)
		pool = mp.Pool(processes=self.NPROC)
		results = [pool.apply_async(call_it,args=(self,func_str,zipped_args[a])) for a in range(arg_size)]
		for a in range(arg_size):
			out[a] = results[a].get()
		pool.close()
		return out
	

