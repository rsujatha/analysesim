import numpy as np
import os
import gc

class Snapshot():
    def __init__(self, snapfile=None, hdf5_support='True',camels=False):
        if camels == True:
            self.paramstring = 'Header'
            self.lengthconversion = 1000. ### length in kpc needs to be converted to mpc
            print ("here i am")
        else:
            print ("or here")
            self.paramstring = 'Parameters'
            self.lengthconversion = 1.
        if hdf5_support:
            import h5py
            self.h5py = h5py
        if snapfile is None:
            print("Instantiated a snapshot object, use 'from_binary' method to read from binary.")
        else:
            self.from_binary(snapfile)



    def from_binary(self, filename = None, header=True):
        assert type(filename) is str, "This class requires the gadget filename as input"
        self.filename = filename
        if not os.path.exists(self.filename): self.filename += '.hdf5' 
        self.filetype = 'gadget_binary' if self.filename[-5:]!='.hdf5' else 'gadget_hdf5'
        if header==True:
            self.read_header()
        else:
            pass

    def read_header(self):
        if self.filetype == 'gadget_binary':
            file = open(self.filename,'rb')
            header_size = np.fromfile(file, dtype=np.uint32, count=1)
            print ("reading the first block (header) which contains ", header_size, " bytes")
            self.N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6)    ## The number of particles of each type present in the file
            self.mass_table       = np.fromfile(file, dtype=np.float64, count=6)     ## Gives the mass of different particles
            self.scale_factor     = np.fromfile(file, dtype=np.float64, count=1)[0]   ##Time of output,  or expansion factor for cosmological simulations
            self.redshift         = np.fromfile(file, dtype=np.float64, count=1)[0]   ## Redshift of the snapshot
            self.flag_sfr         = np.fromfile(file, dtype=np.int32, count=1)[0]     ##Flag for star 
            self.flag_feedback    = np.fromfile(file, dtype=np.int32, count  = 1)[0]  ##Flag for feedback
            self.N_prtcl_total    = np.fromfile(file, dtype=np.uint32, count = 6)  ## Total number of each particle present in the simulation
            self.flag_cooling     = np.fromfile(file, dtype=np.int32, count =1)[0]     ## Flag used for cooling
            self.num_files        = np.fromfile(file, dtype=np.int32, count = 1)[0] ## Number of files in each snapshot
            self.box_size         = np.fromfile(file, dtype = np.float64, count = 1)[0]  ## Gives the box size if periodic boundary conditions are used
            self.Omega_m_0        = np.fromfile(file, dtype = np.float64, count=1)[0]     ## Matter density at z = 0 in the units of critical density
            self.Omega_Lam_0      = np.fromfile(file, dtype = np.float64, count=1)[0]## Vacuum Energy Density at z=0 in the units of critical density
            self.Hubble_param     = np.fromfile(file, dtype = np.float64, count =1 )[0] ## gives the hubble constant in units of 100 kms^-1Mpc^-1  
            self.flag_stellar_age = np.fromfile(file, dtype = np.int32 , count =1)[0]  ##Creation time of stars
            self.flag_metals      = np.fromfile(file, dtype = np.int32 , count =1)[0] ##Flag for metallicity values
            self.N_prtcl_total_HW = np.fromfile(file, dtype = np.int32, count = 6) ## For simulations more that 2^32 particles this field holds the most significant word of the 64 bit total particle number,  otherwise 0
            self.flag_entropy_ICs = np.fromfile(file, dtype = np.int32, count = 1)[0] ## Flag that initial conditions contain entropy instead of thermal energy in the u block
            file.seek(256 +4 , 0)
            header_size_end = np.fromfile(file, dtype = np.int32, count =1)[0]
            print ('Header block is read and it contains ', header_size_end, 'bytes.')
            
        elif self.filetype == 'gadget_hdf5':
            h5file = self.h5py.File(self.filename, 'r')
            self.N_prtcl_thisfile = h5file['Header'].attrs['NumPart_ThisFile']    ## The number of particles of each type present in the file
            self.mass_table       = h5file['Header'].attrs['MassTable']     ## Gives the mass of different particles
            self.scale_factor     = h5file['Header'].attrs['Time']   ##Time of output,  or expansion factor for cosmological simulations
            self.redshift         = h5file['Header'].attrs['Redshift']   ## Redshift of the snapshot
            self.N_prtcl_total    = h5file['Header'].attrs['NumPart_Total']   ## Total number of each particle present in the simulation
            self.num_files        = h5file['Header'].attrs['NumFilesPerSnapshot'] ## Number of files in each snapshot
            self.box_size         = h5file['Header'].attrs['BoxSize']/self.lengthconversion  ## Gives the box size if periodic boundary conditions are used
            self.Omega_m_0        = h5file[self.paramstring].attrs['Omega0']     ## Matter density at z = 0 in the units of critical density
            self.Omega_Lam_0      = h5file[self.paramstring].attrs['OmegaLambda'] ## Vacuum Energy Density at z=0 in the units of critical density
            self.Hubble_param     = h5file[self.paramstring].attrs['HubbleParam'] ## gives the hubble constant in units of 100 kms^-1Mpc^-1  
            # self.num_part_types   = h5file['Config'].attrs['NTYPES']
            self.params           = h5file[self.paramstring].attrs

        self.prtcl_types = ["Gas","Halo","Disk",  "Bulge", "Stars", "Bndry"]
        
    def positions(self, prtcl_type="Halo",max_prtcl=None):
        if self.filetype == 'gadget_binary':
            file = open(self.filename,'rb')
            file.seek(256+8, 0)
            position_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]
            print ("reading the second block (position) which contains ", position_block_size, " bytes")
            i = 0
            while self.prtcl_types[i] != prtcl_type:
                file.seek(self.N_prtcl_thisfile[i]*3*4, 1)
                i += 1
            N_prtcl = self.N_prtcl_thisfile[i] if max_prtcl is None else max_prtcl
            posd = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)  ### The positions are arranged in the binary file as follow: x1,y1,z1,x2,y2,z2,x3,y3,z3 and so on till xn,yn,zn
            posd = posd.reshape((N_prtcl, 3))   ## reshape keeps the fastest changing axis in the end, since x,y,z dimensions are the ones changing the fastest they are given the last axis.
            if max_prtcl is not None:
                print('Positions of {} particles is read'.format(N_prtcl))
            else:
                end  = np.fromfile(file, dtype = np.int32, count =1)[0]
                print ('Position block is read and it contains ', end, 'bytes')
            return posd

        elif self.filetype == 'gadget_hdf5':
            h5file = self.h5py.File(self.filename, 'r')
            # if prtcl_type=="Halo": 
            type_num = self.prtcl_types.index(prtcl_type)
            return h5file[f'PartType{type_num:d}']['Coordinates'][:]/self.lengthconversion

    def velocities(self, prtcl_type="Halo",max_prtcl=None):
        if self.filetype == 'gadget_binary':
            file = open(self.filename,'rb')
            file.seek(256+8+8 + int(self.N_prtcl_thisfile.sum())*3*4, 0)
            velocity_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]
            print ("reading the third block (position) which contains ", velocity_block_size, " bytes")
            i = 0
            while self.prtcl_types[i] != prtcl_type:
                file.seek(self.N_prtcl_thisfile[i]*3*4, 1)
                i += 1
            N_prtcl = self.N_prtcl_thisfile[i] if max_prtcl is None else max_prtcl
            veld = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)  ### The velocities are arranged in the binary file as follow: vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3 and so on till vxn,vyn,vzn
            veld = veld.reshape((N_prtcl, 3))   ## reshape keeps the fastest changing axis in the end, since vx,vy,vz dimensions are the ones changing the fastest they are given the last axis.
            if max_prtcl is not None:
                print('velocities of {} particles is read'.format(N_prtcl))
            else:
                end  = np.fromfile(file, dtype = np.int32, count =1)[0]
                print ('velocity block is read and it contains ', end, 'bytes')
            return veld

        elif self.filetype == 'gadget_hdf5':
            h5file = self.h5py.File(self.filename, 'r')
            # if prtcl_type=="Halo": 
            type_num = self.prtcl_types.index(prtcl_type)
            return h5file[f'PartType{type_num:d}']['Velocities'][:]
          


	# ~ def read_pid(self):
		# ~ ##note: this has not been customised to handle hdf5 files
		# ~ self.g.seek(256 + 8 + 8 + 8 + 2 * self.N_prtcl_thisfile[1]*3*4,0)
		# ~ start_position = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		# ~ print ("reading the fourth block (pID) which contains ",start_position," bytes")
		# ~ pid = np.fromfile(self.g,dtype = np.uint32,count =self.N_prtcl_thisfile[1])
		# ~ end = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		# ~ print ('pID block is read and it contains ',end,'bytes')
		# ~ return pid

	# ~ @np.vectorize
	# ~ def select_these(self,elmt,wanted_elmt):
		# ~ """ Select elements from elmt which belong to wanted_elmt.
			# ~ For fast evaluation with large arrays, ensure wanted_elmt is
			# ~ a set by applying set() to the array.
			# ~ Returns boolean array of shape elmt.
		# ~ """
		# ~ return elmt in wanted_elmt

	# ~ def select_pos_vel(self,wanted_pid):
		# ~ wntd_pid = set(wanted_pid)
		# ~ position =  read_binary_gadget.read_posd(self)
		# ~ velocity =  read_binary_gadget.read_veld(self)
		# ~ pid 	 =  read_binary_gadget.read_pid(self)
		# ~ boolarr  =  read_binary_gadget.select_these(self,pid,wntd_pid)
		# ~ del pid
		# ~ gc.collect()
		# ~ return position[boolarr,:],velocity[boolarr,:]



