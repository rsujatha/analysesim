## This class computes several dark matter halo properties
from __future__ import division
import numpy as np
from intpol	import pmInterpolation
import time
import read_binary_gadget as read
from gadget_tools import Snapshot
import gc
import random
import pickle
from scipy import spatial 
from pathos.multiprocessing import ProcessingPool as Pool
from astropy.stats import jackknife_resampling

class analyse_simulation(object):
	def __init__(self,halofile=None,simfile=None,no_of_subfiles=None,camels=False):
		self.halofile = halofile
		self.simfile = simfile
		self.no_of_subfiles = no_of_subfiles
		self.camels=camels
		if self.simfile is not None:
			if (no_of_subfiles>1):
				print ("greater than 1",simfile+str(0))
#				self.gadget_data = read.read_binary_gadget(simfile+str(0))
				self.gadget_data =Snapshot(simfile+str(0),camels=camels)
			else:
				print ("lesser than 1",simfile)
				self.gadget_data =Snapshot(simfile,camels=camels)
				#self.gadget_data = read.read_binary_gadget(simfile)		
			self.LBox=self.gadget_data.box_size   ##### note: this is in units of h-1 Mpc
			self.ParticleCount = self.gadget_data.N_prtcl_total[1]
		else:
			print ("please make sure to input BoxSize and ParticleCount")

	
	def dn_by_dlnm(self,x,bins):
		## returns number per unit volume (note that volume is in Mpc^3)
		dn = np.histogram(x, bins=bins, normed=False, weights=None, density=None)[0]/(self.LBox)**3
		dlnm = np.mean(np.log(bins[1:]/bins[0:-1]))		
		print (dlnm)
		return dn/dlnm	
	
		
	def deltax(self,GridSize,prtcl_type='Halo'):
		ip = pmInterpolation()
		counts = np.zeros([GridSize,GridSize,GridSize])
		for i in range(self.no_of_subfiles):
			if self.no_of_subfiles==1:
				filename = self.simfile
			else:
				filename = self.simfile + str(i)
			#gadget_data = read.read_binary_gadget(filename)
			gadget_data = Snapshot(filename,camels=self.camels)
			position = gadget_data.positions(prtcl_type=prtcl_type)
			counts += ip.cic(position[:,0],position[:,1],position[:,2],Lbox=self.LBox,GridSize=GridSize)
			del position
			gc.collect()
		return ip.counts_to_overdensity(counts,GridSize,self.ParticleCount)

	def deltax_halo(self,GridSize,x,y,z):
		ip = pmInterpolation()
		return ip.cic(x,y,z,Lbox=self.LBox,GridSize=GridSize,overdensity_flag=1,ParticleCount=np.size(x))

	def deltak_dm(self,GridSize,deltax = None,deconvolve = False):
		if deltax is None:
			deltax = analyse_simulation.deltax(self,GridSize)
		if deconvolve is True:
			k_x, k_y ,k_z =  analyse_simulation.kvector(self,GridSize)
			return (np.fft.rfftn(deltax)/(GridSize)**3) /(np.sinc(self.LBox*k_x/(2*np.pi*GridSize))*np.sinc(self.LBox*k_y/(2*np.pi*GridSize))*np.sinc(self.LBox*k_z/(2*np.pi*GridSize)))**2
		elif deconvolve is False:
			return np.fft.rfftn(deltax)/(GridSize)**3   ### deltak_{our convention} = 1/GridSize**3  deltak_{numpy}(See Aseem's Math Methods Notes and numpy fft documentation)
	
	def deltak_halo(self,GridSize,x,y,z):
		deltax = analyse_simulation.deltax_halo(self,GridSize,x,y,z)
		return np.fft.rfftn(deltax)/(GridSize)**3   ### deltak_{our convention} = 1/GridSize**3  deltak_{numpy}(See Aseem's Math Methods Notes and numpy fft documentation)
	
	def kvector(self,GridSize,flag=0,sparse=False):
		dk = 2*np.pi/self.LBox
		kspace = np.concatenate([range(0,int(GridSize/2)),range(-int(GridSize/2),0)])*dk	
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace[0:int(GridSize/2)+1], indexing='ij',sparse=sparse)
		if flag == 1:
			k_x = np.sin(k_x*self.LBox/GridSize)
			k_y = np.sin(k_y*self.LBox/GridSize)
			k_z = np.sin(k_z*self.LBox/GridSize)
		return k_x,k_y,k_z
	
	def save_deltak_dm(self,GridSize,filepath=None):
		"""
		deltak is arranged in the same order as rfft 
		"""
		if filepath is None:
			filepath = self.simfile
		dk = analyse_simulation.deltak_dm(self,GridSize)
		with open(filepath, 'wb') as fi:
			pickle.dump(dk, fi)
		return

	
	
	
	def deltak_singlehalo(self,x,y,z,GridSize,k_x,k_y,k_z):
		dx = self.LBox / GridSize
		Xvector = np.zeros([8,np.size(x),1,1,1])
		Yvector = np.zeros([8,np.size(x),1,1,1])
		Zvector = np.zeros([8,np.size(x),1,1,1])
		
		weights = np.zeros([8,np.size(x),1,1,1])

		inew = np.mod(np.floor(x/dx),GridSize)
		jnew = np.mod(np.floor(y/dx),GridSize)
		knew = np.mod(np.floor(z/dx),GridSize)
		delx = x - inew*dx
		dely = y - jnew*dx
		delz = z - knew*dx

		#### Cell 0	
		Xvector[0,:,0,0,0] = inew*dx
		Yvector[0,:,0,0,0] = jnew*dx
		Zvector[0,:,0,0,0] = knew*dx
		weights[0,:,0,0,0] = (dx-delx)*(dx-dely)*(dx-delz)/dx**3

		#### Cell 1	
		Xvector[1,:,0,0,0] = np.mod(inew + 1,GridSize)*dx
		Yvector[1,:,0,0,0] = jnew*dx
		Zvector[1,:,0,0,0] = knew*dx
		weights[1,:,0,0,0]   = delx * (dx - dely) * (dx - delz)/dx**3

		
		#### Cell 2	
		Xvector[2,:,0,0,0] = np.mod(inew + 1,GridSize)*dx
		Yvector[2,:,0,0,0] = np.mod(jnew + 1,GridSize)*dx
		Zvector[2,:,0,0,0] = knew*dx
		weights[2,:,0,0,0] = delx * dely * (dx - delz)/dx**3

		
		#### Cell 3	
		Xvector[3,:,0,0,0] = inew*dx
		Yvector[3,:,0,0,0] = np.mod(jnew + 1,GridSize)*dx
		Zvector[3,:,0,0,0] = knew*dx
		weights[3,:,0,0,0]  = (dx - delx)*dely * (dx-delz)/dx**3

		#### Cell 4	
		Xvector[4,:,0,0,0] = inew*dx
		Yvector[4,:,0,0,0] = jnew*dx
		Zvector[4,:,0,0,0] = np.mod(knew + 1,GridSize)*dx
		weights[4,:,0,0,0] =  (dx-delx) * (dx-dely) * delz /dx**3

		#### Cell 5	
		Xvector[5,:,0,0,0] = np.mod(inew + 1,GridSize)*dx
		Yvector[5,:,0,0,0] = jnew*dx
		Zvector[5,:,0,0,0] = np.mod(knew + 1,GridSize)*dx
		weights[5,:,0,0,0] = delx * (dx- dely) * delz /dx**3

		#### Cell 6	
		Xvector[6,:,0,0,0] = np.mod(inew +1,GridSize)*dx
		Yvector[6,:,0,0,0] = np.mod(jnew +1,GridSize)*dx
		Zvector[6,:,0,0,0] = np.mod(knew +1,GridSize)*dx
		weights[6,:,0,0,0] = delx*dely*delz/dx**3

		#### Cell 7	
		Xvector[7,:,0,0,0] = inew*dx
		Yvector[7,:,0,0,0] = np.mod(jnew + 1,GridSize)*dx
		Zvector[7,:,0,0,0] = np.mod(knew + 1,GridSize)*dx
		weights[7,:,0,0,0]  = (dx - delx)*dely*delz/dx**3


		
		exp = np.exp(-1j*( k_x*Xvector + k_y*Yvector + k_z*Zvector ))
		return  np.sum(weights*exp,axis=0)
	

		
	def Pk(self,deltak1,deltak2,GridSize,k_x,k_y,k_z,cic_flag1,cic_flag2):
		if cic_flag1==1:
			deltak1 = deltak1/(np.sinc(self.LBox*k_x/(2*np.pi*GridSize))*np.sinc(self.LBox*k_y/(2*np.pi*GridSize))*np.sinc(self.LBox*k_z/(2*np.pi*GridSize)))**2
		if cic_flag2==1:	
			deltak2 = deltak2/(np.sinc(self.LBox*k_x/(2*np.pi*GridSize))*np.sinc(self.LBox*k_y/(2*np.pi*GridSize))*np.sinc(self.LBox*k_z/(2*np.pi*GridSize)))**2
		return deltak1*deltak2.conjugate()*self.LBox**3 ## P(k) = BoxSize**3 |deltak_{our convention}|**2

		
	def halo_by_halo_bias(self,x,y,z,GridSize,k_x,k_y,k_z,prebins,deltaDarkMatter=None,cic_flag2=1):
		"""
			note:input k_x,k_y,k_z,deltaDarkMatter should be in the order from -kmax to kmax.(Use fftshift to preprocess the fft output) 
			cic_flag2 is set to 1 if the deltaDarkMatter is obtained from a CIC calculation
		"""
		
		n = k_x.shape[2]		
		deltak_halo = analyse_simulation.deltak_singlehalo(self,x,y,z,GridSize,k_x,k_y,k_z)
		if deltaDarkMatter is None:	
			deltak_dm = np.fft.fftshift(analyse_simulation.deltak_dm(self,GridSize),axes=(0,1))[GridSize/2-int(n)+1:GridSize/2+int(n),GridSize/2-int(n)+1:GridSize/2+int(n),0:int(n)]
		else:
			deltak_dm = deltaDarkMatter	
		P_hm = analyse_simulation.Pk(self,deltak_halo,deltak_dm,GridSize,k_x,k_y,k_z,cic_flag1=0,cic_flag2=cic_flag2)
		P_mm = analyse_simulation.Pk(self,deltak_dm,deltak_dm,GridSize,k_x,k_y,k_z,cic_flag1=cic_flag2,cic_flag2=cic_flag2)
				
		P_hm = P_hm.reshape([len(x),np.size(k_x)])
		P_mm = P_mm.reshape([np.size(k_x)])
		k_x = k_x.reshape([np.size(k_x)])
		k_y = k_y.reshape([np.size(k_y)])
		k_z = k_z.reshape([np.size(k_z)])
		
		kvalue = np.sqrt(k_x**2+k_y**2+k_z**2)
		ratio = np.zeros([np.size(x),np.size(prebins)-1])
		for i in range(np.size(prebins)-1):
			cndtn = (kvalue<prebins[i]) | (kvalue>prebins[i+1])
			cndtnn = np.broadcast_to(cndtn,((np.size(x)),cndtn.shape[0]))
			phm = np.mean(np.ma.masked_where(cndtnn,P_hm.real),axis = 1)
			pmm = np.mean(np.ma.masked_where(cndtn,P_mm.real))
			ratio[:,i] = phm/(pmm+1e-15)
		kcube = (np.sqrt(prebins[1:]*prebins[0:-1]))**3.
		b1=np.sum(kcube[None,:]*ratio,axis = 1)/np.sum(kcube)    ####  Average Phm,Pmm in every kbin and weighted averge of it
		cndtn = (kvalue<prebins[0]) | (kvalue>prebins[-1])
		cndtnn = np.broadcast_to(cndtn,((np.size(x)),cndtn.shape[0]))
		phmbypmm = np.mean(np.ma.masked_where(cndtnn,P_hm.real/(P_mm.real+1e-15)),axis = 1) ## Take ratio of Phm/Pmm and finally average it    ### i think highest error
		phmbypmm_v2 = np.mean(np.ma.masked_where(cndtnn,P_hm.real),axis = 1)/(np.mean(np.ma.masked_where(cndtn,P_mm.real))+1e-15)  ## Take average of every mass bin and finally divide Phm/Pmm   ## i think lowest error and is being used in recent places
		return b1,phmbypmm,phmbypmm_v2
		
	def PS_calc(self,deltak1,deltak2,NofParticles=None,NofBins=100,kmin=None,kmax=None,cic_flag=1):
		"""
		Calculates power spectrum P(k) 
		Input: 
		BoxSize - Size of the box 
		deltax  - real space density fluctuation 
		NofBins - number of k bins
		cic_flag - if the deltax is computed using cic then window corrections are done to the power spectrum
		deltak1 - having format of rfft o/p
		deltak2 - having shape of rfft o/p
		Output:
		Pk   -  P(k)  
		k_array -  k in h Mpc^{-1}
		"""
		if NofParticles==None:
			NofParticles = self.ParticleCount
		
		
		
		dk = 2*np.pi/self.LBox
		GridSize = deltak1.shape[0]
		kspace = np.concatenate([range(0,int(GridSize/2)),range(-int(GridSize/2),0)])*dk	
		
		if kmin==None:
			kmin = dk
		if kmax==None:
			kmax = GridSize/2*dk
		
		k_bin = np.logspace(np.log10(kmin),np.log10(kmax),NofBins+1)
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace[0:int(GridSize/2)+1], indexing='ij',sparse=True)
		k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
		### correcting for cic density field 
		if cic_flag==1:
			deltak1 = deltak1/(np.sinc(self.LBox*k_x/(2*np.pi*GridSize))*np.sinc(self.LBox*k_y/(2*np.pi*GridSize))*np.sinc(self.LBox*k_z/(2*np.pi*GridSize)))**2
			deltak2 = deltak2/(np.sinc(self.LBox*k_x/(2*np.pi*GridSize))*np.sinc(self.LBox*k_y/(2*np.pi*GridSize))*np.sinc(self.LBox*k_z/(2*np.pi*GridSize)))**2
		
		Pk = deltak1*deltak2.conjugate()*self.LBox**3 ## P(k) = BoxSize**3 |deltak_{our convention}|**2
		

		Pk = Pk.flatten()
		k = k.flatten()
		### The following line will bin the P(k) logarithmically and average over values belonging to the same bin
		Pk_avg = np.histogram(k , bins = k_bin , weights = Pk)[0]/ (np.histogram(k , bins = k_bin )[0]+1e-15)
		
		#### correcting for shot noise
		Pk_avg -= self.LBox**3/(NofParticles)
		
		
		return Pk_avg, np.sqrt(k_bin[:-1]*k_bin[1:])
		
	# ~ def GaussianWk(self,k,R):
		# ~ ### A normalised gaussian 1/sqrt(2pi)R exp(-k^2R^2/2) 
		# ~ ### R is the standard deviation of the Gaussian.
		# ~ """
		# ~ This function takes standard deviation of a normalised gaussian as input and gives its fourier transform as output
		# ~ """
		# ~ return np.exp(-k**2*R**2/2)
	
	def GaussianWk(self,R,GridSize):
		### A normalised gaussian 1/sqrt(2pi)R exp(-k^2R^2/2) 
		### R is the standard deviation of the Gaussian.
		"""
		This function takes standard deviation of a normalised gaussian as input and gives its fourier transform as output
		"""
		# ~ R = R.reshape(np.size(R),1,1,1)
		# ~ dx = self.LBox / GridSize
		dx = self.LBox / GridSize
		xarray = np.array([range(-int(GridSize/2),int(GridSize/2))])*dx
		x,y,z = np.meshgrid(xarray,xarray,xarray,sparse=True)
		XX = x**2+y**2+z**2
		fn = np.exp(-XX/(2*R**2))
		fn = np.fft.ifftshift(fn)
		kernel= np.fft.rfftn(fn)
		return kernel/kernel[0,0,0]

	def smoothed_deltak(self,R,GridSize,deltak=None,deconvolve = False):
		if deltak is None:
			delk = analyse_simulation.deltak_dm(self,GridSize,deconvolve = deconvolve)
		else:
			delk = deltak
		# ~ k_x, k_y ,k_z =  analyse_simulation.kvector(self,GridSize)
		# ~ k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
		return delk * analyse_simulation.GaussianWk(self,R,GridSize)


	# ~ def TidalTensor(self,GridSize,R):
		# ~ k_x,k_y,k_z = analyse_simulation.kvector(self,GridSize)
		# ~ ksquare = k_x**2 + k_y**2 + k_z**2
		# ~ psik_scalar = - analyse_simulation.smoothed_deltak(self,R,GridSize)/(ksquare+1e-15)
		# ~ TidalTensor = np.zeros([np.size(R),GridSize,GridSize,GridSize,3,3])
		# ~ TidalTensor[:,:,:,:,0,0] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_x ,axes=[-3,-2,-1]))
		# ~ TidalTensor[:,:,:,:,0,1] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_y ,axes=[-3,-2,-1]))
		# ~ TidalTensor[:,:,:,:,0,2] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_z ,axes=[-3,-2,-1]))
		# ~ TidalTensor[:,:,:,:,1,0] = TidalTensor[:,:,:,:,0,1]
		# ~ TidalTensor[:,:,:,:,1,1] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_y * k_y ,axes=[-3,-2,-1]))
		# ~ TidalTensor[:,:,:,:,1,2] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_y * k_z ,axes=[-3,-2,-1]))
		# ~ TidalTensor[:,:,:,:,2,0] = TidalTensor[:,:,:,:,0,2] 
		# ~ TidalTensor[:,:,:,:,2,1] = TidalTensor[:,:,:,:,1,2]
		# ~ TidalTensor[:,:,:,:,2,2] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_z * k_z ,axes=[-3,-2,-1]))
		# ~ return TidalTensor
		
	def TidalTensor(self,GridSize,R,deltak = None,deconvolve=False,flag=1):
		"""
		del^2 psi = delx
		(ik.ik) psik_scalar   = delk 
		psik_scalar = - del_k/k^2
		
		psi_vetor = partial_X partial_Y psi_scalar
		psik_vector = (ik_x)(ik_y) psik_scalar
		
		"""
		k_x,k_y,k_z = analyse_simulation.kvector(self,GridSize,flag=flag,sparse=True)
		ksquare = k_x**2 + k_y**2 + k_z**2
		psik_scalar = - analyse_simulation.smoothed_deltak(self,R,GridSize,deltak = deltak,deconvolve=deconvolve)/(ksquare+1e-15)
		TidalTensor = np.zeros([GridSize,GridSize,GridSize,3,3])
		TidalTensor[:,:,:,0,0] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_x ,axes=[-3,-2,-1]))
		TidalTensor[:,:,:,0,1] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_y ,axes=[-3,-2,-1]))
		TidalTensor[:,:,:,0,2] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_z ,axes=[-3,-2,-1]))
		TidalTensor[:,:,:,1,0] = TidalTensor[:,:,:,0,1]
		TidalTensor[:,:,:,1,1] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_y * k_y ,axes=[-3,-2,-1]))
		TidalTensor[:,:,:,1,2] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_y * k_z ,axes=[-3,-2,-1]))
		TidalTensor[:,:,:,2,0] = TidalTensor[:,:,:,0,2] 
		TidalTensor[:,:,:,2,1] = TidalTensor[:,:,:,1,2]
		TidalTensor[:,:,:,2,2] = (GridSize**3 * np.fft.irfftn( - psik_scalar * k_z * k_z ,axes=[-3,-2,-1]))
		return TidalTensor

	def gridlocator(self,x,y,z,GridSize):
		"""
		given, x,y,z position returns index of the grid location
		"""
		dx = self.LBox / GridSize
		# ~ i_index = np.mod(np.floor(x/dx),GridSize)
		# ~ j_index = np.mod(np.floor(y/dx),GridSize)
		# ~ k_index = np.mod(np.floor(z/dx),GridSize)
		i_index = np.mod(np.round(x/dx),GridSize)
		j_index = np.mod(np.round(y/dx),GridSize)
		k_index = np.mod(np.round(z/dx),GridSize)
		return i_index.astype(int),j_index.astype(int),k_index.astype(int)

	def InterpolatedTidalTensor(self,R1,R2,T1,T2,R_required):
		return (T1-T2)*(R_required-R1)/(R1-R2)+T1
			
	def counter_kdtree_man (self,X,Y,Z,min_r,max_r,bin_num):
		
		'''
		
		
		This function takes the coordinates of the galaxies and returns an array of number of neighbours 
		that fall in a range of 'min_r' to 'max_r' with 'bin_num' number of logarithmic bins. It uses KDTree algorithm 
		for computing it. The code inherently corrects for the edge effects by creating a shell of points assuming
		a periodic boundary condition
		
		
		Input
		------------------
		X					: X Coordinate of the tracer
		Y					: Y Coordinate of the tracer
		Z					: Z Coordinate of the tracer
		min_r				: Minimum radius for obtaining number of neighbour
		max_r				: Maximum radius for obtaining number of neighbour and the Thickness of shell
		bin_num				: Number of logarithmic bins
		
		Output
		------------------
		neighbours			: Number of neighbours in the logarithmically binned shells
		bins				: Bin edges (lower) for the 'neighbours'
		
		'''
		
		
		#~ L=300
		L=self.LBox   ### in Mpch^-1 ,i hope,have to check
		Rmax= max_r
	
		## Box 1 - Right Face 			R
		sel_ind = np.where(X<=Rmax)[0]  
		X1=X[sel_ind]+L
		Y1=Y[sel_ind]
		Z1=Z[sel_ind]
	
	
		## Box 2 - Left Face 			L
		sel_ind = np.where(X>=L-Rmax)[0]  
		X2=X[sel_ind]-L
		Y2=Y[sel_ind]
		Z2=Z[sel_ind]
	
		## Box 3 - Top Face 			T
		sel_ind = np.where(Y<=Rmax)[0]  
		X3=X[sel_ind]
		Y3=Y[sel_ind]+L
		Z3=Z[sel_ind]
	
		## Box 4 - Bottom Face		 	B
		sel_ind = np.where(Y>=L-Rmax)[0]  
		X4=X[sel_ind]
		Y4=Y[sel_ind]-L
		Z4=Z[sel_ind]
	
		## Box 5 - Back Face 			K
		sel_ind = np.where(Z>=L-Rmax)[0]  
		X5=X[sel_ind]
		Y5=Y[sel_ind]
		Z5=Z[sel_ind]-L
	
		## Box 6 - Front Face 			F
		sel_ind = np.where(Z<=Rmax)[0]  
		X6=X[sel_ind]
		Y6=Y[sel_ind]
		Z6=Z[sel_ind]+L
	
		## Box 7 - Edge RT
		sel_ind = np.where((X<=Rmax) & (Y<=Rmax))[0]  
		X7=X[sel_ind]+L
		Y7=Y[sel_ind]+L
		Z7=Z[sel_ind]
	
		## Box 8 - Edge LT
		sel_ind = np.where((X>=L-Rmax)&(Y<=Rmax))[0]  
		X8=X[sel_ind]-L
		Y8=Y[sel_ind]+L
		Z8=Z[sel_ind]
	
		## Box 9 - Edge KT
		sel_ind = np.where((Z>=L-Rmax)&(Y<=Rmax))[0]  
		X9=X[sel_ind]
		Y9=Y[sel_ind]+L
		Z9=Z[sel_ind]-L
	
		## Box 10 -Edge FT
		sel_ind = np.where((Z<=Rmax)&(Y<=Rmax))[0]  
		X10=X[sel_ind]
		Y10=Y[sel_ind]+L
		Z10=Z[sel_ind]+L
	
		## Box 11 - Edge RB
		sel_ind = np.where((X<=Rmax) & (Y>=L-Rmax))[0]  
		X11=X[sel_ind]+L
		Y11=Y[sel_ind]-L
		Z11=Z[sel_ind]
	
		## Box 12 - Edge LB
		sel_ind = np.where((X>=L-Rmax)&(Y>=L-Rmax))[0]  
		X12=X[sel_ind]-L
		Y12=Y[sel_ind]-L
		Z12=Z[sel_ind]
	
		## Box 13 - Edge KB
		sel_ind = np.where((Z>=L-Rmax)&(Y>=L-Rmax))[0]  
		X13=X[sel_ind]
		Y13=Y[sel_ind]-L
		Z13=Z[sel_ind]-L
	
		## Box 14 - Edge FB
		sel_ind = np.where((Z<=Rmax)&(Y>=L-Rmax))[0]  
		X14=X[sel_ind]
		Y14=Y[sel_ind]-L
		Z14=Z[sel_ind]+L
	
		## Box 15 - Edge RF  
		sel_ind = np.where((X<=Rmax)&(Z<=Rmax))[0]  
		X15=X[sel_ind]+L
		Y15=Y[sel_ind]
		Z15=Z[sel_ind]+L
	
		## Box 16 - Edge LF
		sel_ind = np.where((X>=L-Rmax)&(Z<=Rmax))[0]  
		X16=X[sel_ind]-L
		Y16=Y[sel_ind]
		Z16=Z[sel_ind]+L
	
		## Box 17 - Edge RK
		sel_ind = np.where((X<=Rmax)&(Z>=L-Rmax))[0]  
		X17=X[sel_ind]+L
		Y17=Y[sel_ind]
		Z17=Z[sel_ind]-L
	
		## Box 18 - Edge LK
		sel_ind = np.where((X>=L-Rmax)&(Z>=L-Rmax))[0]  
		X18=X[sel_ind]-L
		Y18=Y[sel_ind]
		Z18=Z[sel_ind]-L
	
		## Box 19 - Corner RTK
		sel_ind = np.where((X<=Rmax)&(Y<=Rmax)&(Z>=L-Rmax))[0]  
		X19=X[sel_ind]+L
		Y19=Y[sel_ind]+L
		Z19=Z[sel_ind]-L 
	
		## Box 20 - Corner RBK
		sel_ind = np.where((X<=Rmax)&(Y>=L-Rmax)&(Z>=L-Rmax))[0]  
		X20=X[sel_ind]+L
		Y20=Y[sel_ind]-L
		Z20=Z[sel_ind]-L
	
		## Box 21 - Corner RTF
		sel_ind = np.where((X<=Rmax)&(Y<=Rmax)&(Z<=Rmax))[0]  
		X21=X[sel_ind]+L
		Y21=Y[sel_ind]+L
		Z21=Z[sel_ind]+L
	
		## Box 22 - Corner RBF
		sel_ind = np.where((X<=Rmax)&(Y>=L-Rmax)&(Z<=Rmax))[0]  
		X22=X[sel_ind]+L
		Y22=Y[sel_ind]-L
		Z22=Z[sel_ind]+L
	
		## Box 23 - Corner LTK
		sel_ind = np.where((X>=L-Rmax)&(Y<=Rmax)&(Z>=L-Rmax))[0]  
		X23=X[sel_ind]-L
		Y23=Y[sel_ind]+L
		Z23=Z[sel_ind]-L 
	
		## Box 24 - Corner LBK
		sel_ind = np.where((X>=L-Rmax)&(Y>=L-Rmax)&(Z>=L-Rmax))[0]  
		X24=X[sel_ind]-L
		Y24=Y[sel_ind]-L
		Z24=Z[sel_ind]-L
	
		## Box 25 - Corner LTF
		sel_ind = np.where((X>=L-Rmax)&(Y<=Rmax)&(Z<=Rmax))[0]  
		X25=X[sel_ind]-L
		Y25=Y[sel_ind]+L
		Z25=Z[sel_ind]+L
	
		## Box 26 - Corner LBF
		sel_ind = np.where((X>=L-Rmax)&(Y>=L-Rmax)&(Z<=Rmax))[0]  
		X26=X[sel_ind]-L
		Y26=Y[sel_ind]-L
		Z26=Z[sel_ind]+L
	
	
		X_shell = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26))
		Y_shell = np.concatenate((Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10,Y11,Y12,Y13,Y14,Y15,Y16,Y17,Y18,Y19,Y20,Y21,Y22,Y23,Y24,Y25,Y26))
		Z_shell = np.concatenate((Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11,Z12,Z13,Z14,Z15,Z16,Z17,Z18,Z19,Z20,Z21,Z22,Z23,Z24,Z25,Z26))
	
	
		X_all = np.concatenate((X,X_shell))
		Y_all = np.concatenate((Y,Y_shell))
		Z_all = np.concatenate((Z,Z_shell))
	
		bins = 10**(np.linspace(np.log10(min_r),np.log10(max_r),bin_num))
		pts_all=np.transpose(np.array([X_all,Y_all,Z_all]))
		pts=np.transpose(np.array([X,Y,Z]))
		T = spatial.cKDTree(pts)
		T_all=spatial.cKDTree(pts_all)
		n=T.count_neighbors(T_all,bins)
		neighbours = ( n[1:]-n[:-1])/2
		return  neighbours, bins

			
	def counter_kdtree (self,X,Y,Z,min_r,max_r,bin_num,jacknife=0):
		
		'''
		This function takes the coordinates of the galaxies and returns an array of number of neighbours 
		that fall in a range of 'min_r' to 'max_r' with 'bin_num' number of logarithmic bins. It uses KDTree algorithm 
		for computing it. The code inherently corrects for the edge effects by creating a shell of points assuming
		a periodic boundary condition
		
		
		Input
		------------------
		X					: X Coordinate of the tracer
		Y					: Y Coordinate of the tracer
		Z					: Z Coordinate of the tracer
		min_r				: Minimum radius for obtaining number of neighbour
		max_r				: Maximum radius for obtaining number of neighbour and the Thickness of shell
		bin_num				: Number of logarithmic bins
		jacknife            : Number of jacknife slices the simulation is divided along each dimension. if 0, then no jacknife estimates are outputd
		
		Output
		------------------
		neighbours			: Number of neighbours in the logarithmically binned shells
		bins				: Bin edges (lower) for the 'neighbours'
		
		'''
		
		
		bins = 10**(np.linspace(np.log10(min_r),np.log10(max_r),bin_num))
		pts=np.transpose(np.array([X,Y,Z]))
		T = spatial.cKDTree(pts,boxsize=self.LBox) ### this supposedly takes care of periodic boundary conditions
		n=T.count_neighbors(T,bins)
		neighbours = ( n[1:]-n[:-1])/2
		if jacknife==0:
			return  neighbours, bins
		elif jacknife=='delete-1':
			covn = np.empty([len(bins)-1,len(X)])
			for i in range(len(X)):
				cond = np.full(len(X),True)
				cond[i] = False
				njack  = T.count_neighbors(T,bins,weights=cond)
				covn[:,i] = ( njack[1:]-njack[:-1])/2
				pairsN=float(cond.sum()*(cond.sum()-1)/2.)
				covn[:,i]=covn[:,i]/pairsN
				rr=bins[1:]**3-bins[:-1]**3
				RR = 4/3.*np.pi*(rr)/(self.LBox**3*(1-1/len(X)))
				covn[:,i] = covn[:,i]/RR -1
			return neighbours,bins,covn
		else:
			delB = self.LBox/jacknife
			no_of_procs = 10
			tbegin=time.time()
			def computexir(Noj):
				i = Noj//jacknife**2
				j = (Noj-i*jacknife**2)//jacknife
				k = (Noj-i*jacknife**2-j*jacknife)
				cond = ((X<i*delB)|(X>(i+1)*delB)|(Y<j*delB)|(Y>(j+1)*delB)|(Z<k*delB)|(Z>(k+1)*delB)).astype(int)
				njack  = T.count_neighbors(T,bins,weights=cond)
				covn = ( njack[1:]-njack[:-1])/2
				pairsN=float(cond.sum()*(cond.sum()-1)/2.)
				covn=covn/pairsN
				rr=bins[1:]**3-bins[:-1]**3
				RR = 4/3.*np.pi*(rr)/(self.LBox**3*(1-1/jacknife**3))
				covn=covn/RR-1		
				return  covn		
			p = Pool(no_of_procs,maxtasksperchild=1)
			covn = p.map(computexir, range(jacknife**3))
			#p.close()
			#p.join()
			covn = np.array(covn)
			print ("timetaken",time.time()-tbegin)
			return neighbours,bins,covn.T
			
	def LS_Xir(self,X,Y,Z,min_r,max_r,bin_num,jacknife=0):
		"""
		cov_jacknife = 0 default, any other number specifies the number of pieces to divide the box to compute covariance matrix
		"""
		if jacknife==0:
			DD,bins=analyse_simulation.counter_kdtree(self,X,Y,Z,min_r,max_r,bin_num)
			pairsN=float(X.size*(X.size-1)/2.)
			DD=DD/pairsN
			rr=bins[1:]**3-bins[:-1]**3
			RR = 4/3.*np.pi*(rr)/self.LBox**3
			Xi=DD/RR-1
			r=(bins[1:]+bins[:-1])/2.
			return Xi,r
		else:
			DD,bins,cov=analyse_simulation.counter_kdtree(self,X,Y,Z,min_r,max_r,bin_num,jacknife)
			pairsN=float(X.size*(X.size-1)/2.)
			DD=DD/pairsN
			rr=bins[1:]**3-bins[:-1]**3
			RR = 4/3.*np.pi*(rr)/self.LBox**3
			Xi=DD/RR-1
			r=(bins[1:]+bins[:-1])/2.		
			return Xi,r,cov
			
	def cross_counter_kdtree (self,X1,Y1,Z1,X2,Y2,Z2,min_r,max_r,bin_num,jacknife=0):
		
		'''
		This function takes the coordinates of the galaxies and returns an array of number of neighbours 
		that fall in a range of 'min_r' to 'max_r' with 'bin_num' number of logarithmic bins. It uses KDTree algorithm 
		for computing it. The code inherently corrects for the edge effects by creating a shell of points assuming
		a periodic boundary condition
		
		
		Input
		------------------
		X1					: X Coordinate of the tracer1
		Y1					: Y Coordinate of the tracer1
		Z1					: Z Coordinate of the tracer1
		X2					: X Coordinate of the tracer2
		Y2					: Y Coordinate of the tracer2
		Z2					: Z Coordinate of the tracer2
		min_r				: Minimum radius for obtaining number of neighbour
		max_r				: Maximum radius for obtaining number of neighbour and the Thickness of shell
		bin_num				: Number of logarithmic bins
		jacknife            : Number of jacknife slices the simulation is divided along each dimension. if 0, then no jacknife estimates are outputd
		
		Output
		------------------
		neighbours			: Number of neighbours in the logarithmically binned shells
		bins				: Bin edges (lower) for the 'neighbours'
		
		'''
		
		
		bins = 10**(np.linspace(np.log10(min_r),np.log10(max_r),bin_num))
		pts1=np.transpose(np.array([X1,Y1,Z1]))
		pts2=np.transpose(np.array([X2,Y2,Z2]))
		T1 = spatial.cKDTree(pts1,boxsize=self.LBox) ### this supposedly takes care of periodic boundary conditions
		T2 = spatial.cKDTree(pts2,boxsize=self.LBox) ### this supposedly takes care of periodic boundary conditions
		n=T1.count_neighbors(T2,bins)
		neighbours = ( n[1:]-n[:-1])
		if jacknife==0:
			return  neighbours, bins
		else:
			delB = self.LBox/jacknife
			def computexir(Noj):
				i = Noj//jacknife**2
				j = (Noj-i*jacknife**2)//jacknife
				k = (Noj-i*jacknife**2-j*jacknife)
				cond1= ((X1<i*delB)|(X1>(i+1)*delB)|(Y1<j*delB)|(Y1>(j+1)*delB)|(Z1<k*delB)|(Z1>(k+1)*delB)).astype(int)
				cond2= ((X2<i*delB)|(X2>(i+1)*delB)|(Y2<j*delB)|(Y2>(j+1)*delB)|(Z2<k*delB)|(Z2>(k+1)*delB)).astype(int)
				njack  = T1.count_neighbors(T2,bins,weights=(cond1,None))
				covn = ( njack[1:]-njack[:-1])
				pairsN=float(cond1.sum()*(cond2.sum()))
				covn=covn/pairsN
				rr=bins[1:]**3-bins[:-1]**3
				RR = 4/3.*np.pi*(rr)/(self.LBox**3*(1-1/jacknife**3))
				covn=covn/RR-1
				return  covn	
			no_of_procs = 6
			p = Pool(no_of_procs,maxtasksperchild=1)
			covn = p.map(computexir, range(jacknife**3))
			# ~ covn = np.empty([jacknife**3,len(bins)-1])
			# ~ for i in range(jacknife**3):
				# ~ covn[i] =  computexir(i)
			covn = np.array(covn)
			return neighbours,bins,covn.T
			#
			#
			#
			#tbegin=time.time()
			#print ('beg')
			#def computexir(Noj):
			#	i = Noj//jacknife**2
			#	j = (Noj-i*jacknife**2)//jacknife
			#	k = (Noj-i*jacknife**2-j*jacknife)
			#	cond1= ((X1<i*delB)|(X1>(i+1)*delB)|(Y1<j*delB)|(Y1>(j+1)*delB)|(Z1<k*delB)|(Z1>(k+1)*delB)).astype(int)
			#	cond2= ((X2<i*delB)|(X2>(i+1)*delB)|(Y2<j*delB)|(Y2>(j+1)*delB)|(Z2<k*delB)|(Z2>(k+1)*delB)).astype(int)
			#	njack  = T1.count_neighbors(T2,bins,weights=tuple([cond1,cond2]))
			#	covn = ( njack[1:]-njack[:-1])
			#	pairsN=float(cond1.sum()*(cond2.sum()))
			#	covn=covn/pairsN
			#	rr=bins[1:]**3-bins[:-1]**3
			#	RR = 4/3.*np.pi*(rr)/(self.LBox**3*(1-1/jacknife**3))
			#	covn=covn/RR-1
			#	print ("here")		
			#	return  covn		
			##p = Pool(no_of_procs,maxtasksperchild=1)
			##covn = p.map(computexir, range(jacknife**3))
			#covn = np.empty([jacknife**3,len(bins)-1])
			#for i in range(jacknife**3):
			#	covn[i] =  computexir(i)
			##p.close()
			##p.join()
			##covn = np.array(covn)
			#print ("timetaken",time.time()-tbegin)
			#return neighbours,bins,covn.T
			
	def cross_Xir(self,X1,Y1,Z1,X2,Y2,Z2,min_r,max_r,bin_num,jacknife=0):
		"""
		cov_jacknife = 0 default, any other number specifies the number of pieces to divide the box to compute covariance matrix
		"""
		if jacknife==0:
			DD,bins=analyse_simulation.cross_counter_kdtree(self,X1,Y1,Z1,X2,Y2,Z2,min_r,max_r,bin_num)
			pairsN=float(X1.size*(X2.size))
			DD=DD/pairsN
			rr=bins[1:]**3-bins[:-1]**3
			RR = 4/3.*np.pi*(rr)/self.LBox**3
			Xi=DD/RR-1
			r=(bins[1:]+bins[:-1])/2.
			return Xi,r
		else:
			DD,bins,cov=analyse_simulation.cross_counter_kdtree(self,X1,Y1,Z1,X2,Y2,Z2,min_r,max_r,bin_num,jacknife)
			pairsN=float(X1.size*(X2.size))
			DD=DD/pairsN
			rr=bins[1:]**3-bins[:-1]**3
			RR = 4/3.*np.pi*(rr)/self.LBox**3
			Xi=DD/RR-1
			r=(bins[1:]+bins[:-1])/2.		
			return Xi,r,cov

	def overdensity_to_particleprobdensity(self,overdensity):
		"""
		takes as input the overdensity and gives as outputs the probability for sampling particles from each grid
		n - no of particles in a grid
		nbar - avg no of particles in a grid
		ntotal - total no of particles
		NGrid - total no of grids 
		n = nbar (1+delta)
		particle probability density = n/ntotal = ntotal/Ngrid*(1+delta)/ntotal = (1+delta)/NGrid
		"""
		Nd = len(overdensity)
		n = (1+overdensity)/Nd**3 
		n/=n.sum()
		return n

	def sample_particles_from_density(self,deltax,samplesize):
		"""
		input:
		deltax --> overdensity of the simulation box
		samplesize --> particle sampling required
		
		output:
		xsample,ysample,zsample - > xyz positions of the particles that are sampled from the density field
		"""
		Nd = len(deltax)
		prob = self.overdensity_to_particleprobdensity(deltax)
		
		ind = np.mgrid[0:Nd,0:Nd,0:Nd]
		nbin = ind[0]*Nd**2+ind[1]*Nd+ind[2]
		
		nbinsample = np.random.choice(nbin.ravel(order='C'),size=samplesize,p=prob.ravel(order='C'))
			
		ixsample = nbinsample//Nd**2
		iysample = (nbinsample - ixsample*Nd**2)//Nd
		izsample = nbinsample - ixsample*Nd**2 - iysample*Nd
		
		print (ixsample.max(),ixsample.min())
		xsample = (ixsample + np.random.uniform(low=0,high=1,size=samplesize))*self.LBox/Nd
		ysample = (iysample + np.random.uniform(low=0,high=1,size=samplesize))*self.LBox/Nd
		zsample = (izsample + np.random.uniform(low=0,high=1,size=samplesize))*self.LBox/Nd
		
		return xsample,ysample,zsample
