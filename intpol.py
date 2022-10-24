### created by Sujatha Ramakrishnan  

from __future__ import division
import numpy as np
class pmInterpolation(object):
		
	def Counts(self,BinNumber,GridSize,Weight):
		if Weight is not None:
			Weightn = Weight.flatten('C')
		else: Weightn = Weight
		Counts = np.bincount((BinNumber.astype(int)).flatten(order='C'),Weightn)
		padcount = GridSize**3-Counts.shape[0] 
		Counts = np.pad(Counts,(0,padcount),'constant')
		return Counts.reshape([GridSize,GridSize,GridSize])
		
	def ksquare(self,GridSize,LBox):
		dk = 2*np.pi/(LBox)
		kspace = np.concatenate([range(0,int(GridSize/2)),range(-int(GridSize/2),0)])*dk	
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace, indexing='ij')
		ksquare = k_x**2 + k_y**2 + k_z**2
		return ksquare
			
	def ngp(self,PositionX,PositionY,PositionZ,Lbox,GridSize):
		dx = Lbox / GridSize
		# ~ inew = np.floor(np.mod(PositionX + dx/2,Lbox)/dx)
		# ~ jnew = np.floor(np.mod(PositionY + dx/2,Lbox)/dx)
		# ~ knew = np.floor(np.mod(PositionZ + dx/2,Lbox)/dx)
		inew = np.mod(np.floor(PositionX/dx),GridSize)
		jnew = np.mod(np.floor(PositionY/dx),GridSize)
		knew = np.mod(np.floor(PositionZ/dx),GridSize)
		BinNumber = inew*GridSize**2 + GridSize *jnew + knew  
		return pmInterpolation.Counts(self,BinNumber,GridSize,Weight=None)
		
	def ngp_smooth(self,PositionX,PositionY,PositionZ,Lbox,GridSize,R):
		ksquare = pmInterpolation.ksquare(self,GridSize,Lbox)
		TotalCounts = pmInterpolation.ngp(self,PositionX,PositionY,PositionZ,Lbox,GridSize)     
		TotalCountsk = np.fft.fftn(TotalCounts)	
		TotalCounts_smooth = np.fft.ifftn(TotalCountsk*pmInterpolation.GaussianWk(self,np.sqrt(ksquare),R)).real
		return TotalCounts_smooth
		
	def cic_smooth(self,PositionX,PositionY,PositionZ,Lbox,GridSize,R):
		ksquare = pmInterpolation.ksquare(self,GridSize,Lbox)
		TotalCounts = pmInterpolation.cic(self,PositionX,PositionY,PositionZ,Lbox,GridSize)     
		TotalCountsk = np.fft.fftn(TotalCounts)	
		TotalCounts_smooth = np.fft.ifftn(TotalCountsk*pmInterpolation.GaussianWk(self,np.sqrt(ksquare),R)).real
		return TotalCounts_smooth
		
	def cic(self,PositionX,PositionY,PositionZ,Lbox,GridSize,overdensity_flag=0,ParticleCount=None):
		###in case of overdensity calculation please provide total particle count and not cubs root of it
		# ~ dx = Lbox / GridSize
		#### Cell 0	
		inew = np.mod(np.floor(PositionX*GridSize/Lbox),GridSize)
		jnew = np.mod(np.floor(PositionY*GridSize/Lbox),GridSize)
		knew = np.mod(np.floor(PositionZ*GridSize/Lbox),GridSize)
		BinNumber = inew*GridSize**2  + GridSize *jnew + knew
		# ~ delx = PositionX - inew*dx -dx/2.0
		# ~ dely = PositionY - jnew*dx -dx/2.0
		# ~ delz = PositionZ - knew*dx -dx/2.0
		delx = PositionX*GridSize/Lbox - inew 
		dely = PositionY*GridSize/Lbox - jnew 
		delz = PositionZ*GridSize/Lbox - knew 
		Weight = (1-delx)*(1-dely)*(1-delz)
		TotalCounts = pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
		
		#### Cell 1	
		BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *jnew + knew
		Weight	   = delx * (1 - dely) * (1 - delz)
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
		
		
		#### Cell 2	
		BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + knew
		Weight	   = delx * dely * (1 - delz)
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
	 
		#### Cell 3	
		BinNumber = inew*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + knew
		Weight	  = (1 - delx)*dely * (1-delz)
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
		#~ #### Cell 4	
		BinNumber = inew*GridSize**2  + GridSize *jnew + np.mod(knew + 1,GridSize)
		Weight	  =  (1-delx) * (1-dely) * delz 
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
 
		#~ #### Cell 5	
		BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *jnew + np.mod(knew + 1,GridSize)
		Weight	  = delx * (1- dely) * delz 
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
		#### Cell 6	
		BinNumber = np.mod(inew +1,GridSize)*GridSize**2  + GridSize *np.mod(jnew +1,GridSize) + np.mod(knew + 1,GridSize)
		Weight	  = delx*dely*delz
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
		#~ #### Cell 7	
		BinNumber = inew*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + np.mod(knew + 1,GridSize)
		Weight	  = (1 - delx)*dely*delz
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
		print (np.sum(TotalCounts))
		if overdensity_flag==0:
			return TotalCounts
		elif overdensity_flag==1:
			return TotalCounts*GridSize**3/ParticleCount - 1

	def cicl(self,PositionX,PositionY,PositionZ,Lbox,GridSize):
		dx = Lbox / GridSize
		Counts = np.zeros([GridSize,GridSize,GridSize])
		for i in range(0,PositionX.shape[0]):
			inew  = int(np.mod(np.floor(PositionX[i]/dx),GridSize))
			inew1 = int(np.mod(inew + 1,GridSize))
			jnew  = int(np.mod(np.floor(PositionY[i]/dx),GridSize))
			jnew1 = int(np.mod(jnew + 1,GridSize))
			knew  = int(np.mod(np.floor(PositionZ[i]/dx),GridSize))
			knew1 = int(np.mod(knew + 1,GridSize))
			delx  = PositionX[i] - inew*dx
			dely  = PositionY[i] - jnew*dx
			delz  = PositionZ[i] - knew*dx
			
			Counts[inew,jnew,knew] += (dx-delx)*(dx-dely)*(dx-delz)/dx**3
			Counts[inew1,jnew,knew] += delx * (dx- dely) * (dx - delz) / dx**3 
			Counts[inew1,jnew1,knew] +=delx * dely * (dx - delz)/dx**3
			Counts[inew,jnew1,knew] += (dx - delx)*dely * (dx-delz)/dx**3
			Counts[inew,jnew,knew1] += (dx-delx) * (dx-dely) * delz /dx**3
			Counts[inew1,jnew,knew1] +=  delx * (dx- dely) * delz /dx**3
			Counts[inew1,jnew1,knew1] += delx*dely*delz/dx**3
			Counts[inew,jnew1,knew1] += (dx - delx)*dely*delz/dx**3
		return Counts
		
	def GaussianWk(self,k,R):
		### R is the standard deviation of the Gaussian.
		return np.exp(-k**2*R**2/2)

	def counts_to_overdensity(self,Counts,GridSize,ParticleCount):
		return Counts * GridSize**3/ParticleCount - 1
