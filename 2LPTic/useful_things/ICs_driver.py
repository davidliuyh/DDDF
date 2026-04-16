# We use this script to create the initial condition parameter file when multiple
# realizations of the same cosmology are needed
import numpy as np
import sys,os

############################### INPUT ###################################
fiducial_file = '2LPT.param' #fiducial file

realizations  = 1000 #number of realizations

fiducial_seed = 7890 #Seed value in the fiducial file
fiducial_flip = 4567 #Phase_flip value in the fiducial file
fiducial_RS   = 3456 #RayleighSampling value in the fiducial file 
#########################################################################

# do a loop over all realizations
for i in xrange(realizations):

    seed = 10*i + 5

    # create the folder if it does not exists
    folder = str(i)
    if not(os.path.exists(folder)):  os.system('mkdir %s'%folder)

    # open input and output files
    fin  = open(fiducial_file, 'r')
    fout = open(folder+'/%s'%fiducial_file, 'w')

    for line in fin:
        if str(fiducial_seed) in line.split():
            fout.write(line.replace(str(fiducial_seed), str(seed)))
        elif str(fiducial_RS) in line.split():
            fout.write(line.replace(str(fiducial_RS), str(1)))  #do Rayleigh sampling
        elif str(fiducial_flip) in line.split():
            fout.write(line.replace(str(fiducial_flip), str(0)))
        else:  fout.write(line)
        
    fin.close(); fout.close()



