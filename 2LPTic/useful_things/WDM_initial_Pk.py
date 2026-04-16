#This script computes the initial P(k) for a WDM model and save the results
#to a file. The input is the CAMB matter power spectrum, the value of
#Omega_m, h and the mass of the WDM particle
import numpy as np

################################## INPUT #####################################
f_in='CAMB_TABLES/ics_matterpow_99.dat'

Omega_m=0.3175
h=0.6711
m_WDM=2.0 #keV

f_out='Pk_WDM_z=99.dat'
##############################################################################

#read CAMB P(k) file
k,Pk=np.loadtxt(f_in,unpack=True)

#compute the WDM P(k) (see Viel et al. 2011: 1107.4094)
nu=1.12
alpha = 0.049*(1.0/m_WDM)**(1.11)*(Omega_m/0.25)**(0.11)*(h/0.7)**(1.22) #Mpc/h
Pk_WDM=(1.0+(alpha*k)**(2.0*nu))**(-5.0/nu)*Pk

#save results to file
np.savetxt(f_out,np.transpose([k,Pk_WDM]))
