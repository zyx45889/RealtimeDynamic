# data_gen
from manta import *
import os, shutil, math, sys, time
from datetime import datetime
import numpy as np
import h5py
import time
import datetime

save_intvel=3

def gen_data(dataid):
    start_time=datetime.datetime.now()
    steps = 60
    stop_step = 40
    basePath = './data/'+str(dataid)
    if not os.path.exists(basePath):
        os.mkdir(basePath)
    res = 128
    blurSig = float(2) / 3.544908
    dim = 3
    timeOffset = 0
    npSeed = np.random.randint(0, 7121708 )
    np.random.seed(npSeed)

    # Init solvers -------------------------------------------------------------------#
    grid_scale = vec3(res,res,res)

    buoyFac = np.random.rand() + 0.1
    velFac = 1.0
    scaleFactor=2 # 4
    buoy    = vec3(0,-3e-3,0) * buoyFac* vec3(1./scaleFactor)

    # solvers
    mysolver = Solver(name='main', gridSize = grid_scale, dim=dim)
    mysolver.timestep=2

    # Simulation Grids  -------------------------------------------------------------------#
    flags   = mysolver.create(FlagGrid)
    vel     = mysolver.create(MACGrid)
    density = mysolver.create(RealGrid)
    tmp     = mysolver.create(RealGrid)
    density_tmp = mysolver.create(RealGrid)
    velRecenter    = mysolver.create(MACGrid)

    # open boundaries
    bWidth=1
    flags.initDomain(boundaryWidth=bWidth)
    flags.fillGrid()

    setOpenBound(flags,    bWidth,'yY',FlagOutflow|FlagEmpty) 

    # wavelet turbulence octaves
    wltnoise = NoiseField( parent=mysolver, loadFromFile=True)
    # scale according to lowres sim , smaller numbers mean larger vortices
    wltnoise.posScale = vec3( int(1.0*grid_scale.x/scaleFactor) ) * 0.5
    wltnoise.timeAnim = 0.1

    wltnoise2 = NoiseField( parent=mysolver, loadFromFile=True)
    wltnoise2.posScale = wltnoise.posScale * 2.0
    wltnoise2.timeAnim = 0.1

    wltnoise3 = NoiseField( parent=mysolver, loadFromFile=True)
    wltnoise3.posScale = wltnoise2.posScale * 2.0
    wltnoise3.timeAnim = 0.1

    # inflow sources ----------------------------------------------------------------------#

    # init random density
    sources  = []
    source_iniv=[]
    source_iniv_vy=[]
    noise    = []
    inflowSrc = [] # list of IDs to use as continuous density inflows

    noiseN = 1+np.random.randint(5)
    nseeds = np.random.randint(10000,size=noiseN)

    cpos = vec3(0.5,0.5,0.5)

    randoms = np.random.rand(noiseN, 10)

    for nI in range(noiseN):
        noise.append( mysolver.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
        noise[nI].posScale = vec3( res * 0.1 * (randoms[nI][7])+1.0) * ( float(scaleFactor))
        noise[nI].clamp = True
        noise[nI].clampNeg = 0
        noise[nI].clampPos = 1.0
        noise[nI].valScale = 1.0
        noise[nI].valOffset = 0.5 * randoms[nI][9]
        noise[nI].timeAnim = 0.3
        noise[nI].posOffset = vec3(1.5)
        
        # random offsets
        coff = vec3(0.4) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
        radius_rand = 0.035 + 0.035 * randoms[nI][3]
        # while radius_rand < 0.04:
        # 	radius_rand *= 1.1
        # while radius_rand > 0.45:
        # 	radius_rand /= 1.1
        upz = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )

        if 1 :#and randoms[nI][8] > 0.5: 
            if coff.y > -0.2:
                coff.y += -0.4
            coff.y *= 0.5
            coff.y=-0.40
            inflowSrc.append(nI)

        sources.append(mysolver.create(Sphere, center=grid_scale*(cpos+coff), radius=grid_scale.x*radius_rand, scale=upz))
        source_iniv.append(mysolver.create(Sphere, center=grid_scale*(cpos+coff), radius=grid_scale.x*radius_rand, scale=upz))
        # s.create(Cylinder, center=gs*vec3(0.5,0.1,0.5), radius=res*0.14, z=gs*vec3(0, 0.02, 0))
            
        print (nI, "centre", grid_scale*(cpos+coff), "radius", grid_scale.x*radius_rand, "other", upz )	
        densityInflow( flags=flags, density=density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
        source_iniv_vy.append(1.5+np.random.rand())
        source_iniv[nI].applyToGrid(grid=vel , value=vec3(0,source_iniv_vy[nI],0) )

    # init random velocities

    inivel_sources = []
    inivel_vels = []
    c = 3 + np.random.randint(3) # "sub" mode

    # 3..5 - ini vel sources
    if c==3: 
        numFac = 1
        sizeFac = 0.9
    if c==4: 
        numFac = 3
        sizeFac = 0.7
    if c==5: 
        numFac = 5
        sizeFac = 0.6
    numNs = int( numFac * float(dim) )
    for ns in range(numNs):
        p = [0.5,0.5,0.5]
        Vrand = np.random.rand(10) 
        for i in range(3):
            p[i] += (Vrand[0+i]-0.5) * 0.6
        p = Vec3(p[0],p[1],p[2])
        size = ( 0.05 + 0.1*Vrand[3] ) * sizeFac

        v = [0.,0.,0.]
        for i in range(3):
            v[i] -= (Vrand[0+i]-0.5) * 0.6 * 2. # invert pos offset , towards middle
            v[i] += (Vrand[4+i]-0.5) * 0.3      # randomize a bit, parametrized for 64 base
        v = Vec3(v[0],v[1],v[2])
        v = v*0.9 # tweaking
        v = v*(1. + 0.5*Vrand[7] ) # increase by up to 50% 
        v *= float(scaleFactor)

        sourceV = mysolver.create(Sphere, center=grid_scale*p, radius=grid_scale.x*size, scale=vec3(1))
        inivel_sources.append(sourceV)
        inivel_vels.append(v)

    def save_data(filename,density):

        f=h5py.File(filename,"w")
        f.create_dataset('data/tomography/data',data=density)
        
        f.close()

    doRecenter=False

    def calcCOM(dens):
        if doRecenter:
            newCentre = calcCenterOfMass(density)
            velOffset = grid_scale*float(0.5) - newCentre
            velOffset = velOffset * (1./ mysolver.timestep)
        else: 
            velOffset = vec3(0.0) # re-centering off

        return velOffset
    
    end_time = datetime.datetime.now()
    print("initialization time: ", end_time-start_time)

    t = 0
    # main loop --------------------------------------------------------------------#
    while t < steps + timeOffset:
        start_time = datetime.datetime.now()
        curt = t * mysolver.timestep
        print( "Current sim time t: " + str(curt) +" \n" )

        velOffset = calcCOM(density)

        if 1 and len(inflowSrc)>0 and t<stop_step:
            # note - the density inflows currently move with the offsets!
            for nI in inflowSrc:
                densityInflow( flags=flags, density=density, noise=noise[nI], shape=sources[nI], scale=1.0*(stop_step-t)/stop_step, sigma=1.0 )
                source_iniv[nI].applyToGrid(grid=vel , value=vec3(0,source_iniv_vy[nI],0) )

        advectSemiLagrange(flags=flags, vel=velRecenter, grid=vel, order=2, clampMode=2)
        setWallBcs(flags=flags, vel=vel)
        addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)
        for i in range(len(inivel_sources)):
            inivel_sources[i].applyToGrid( grid=vel , value=inivel_vels[i] )
        if 1 and ( t< timeOffset ): 
            vorticityConfinement( vel=vel, flags=flags, strength=0.05 )

        solvePressure(flags=flags, vel=vel, pressure=tmp ,  cgMaxIterFac=2.0, cgAccuracy=0.001, preconditioner=PcMGStatic )
        setWallBcs(flags=flags, vel=vel)
        velRecenter.copyFrom( vel )
        velRecenter.addConst( velOffset )
        advectSemiLagrange(flags=flags, vel=velRecenter, grid=density, order=2, clampMode=2)

        # save all frames
        if t>=timeOffset and (t-timeOffset+1)%save_intvel == 0:
            tf = (t-timeOffset)/save_intvel
            print("Writing NPZs for frame %d"%tf)
            
            density_np = np.zeros([int(grid_scale.z), int(grid_scale.y), int(grid_scale.x), 1])
            copyGridToArrayReal( target=density_np, source=density )
            save_data(basePath+'/density_%04d.emd' % (tf),density_np[:,:,:,0])
            # np.savez_compressed(basePath+'/density_%04d.npz' % (tf),data=density_np[:,:,:,0])

        mysolver.step()
        t = t+1
        end_time = datetime.datetime.now()
        print("step time: ", end_time-start_time)

datanum=200
for i in range(0,datanum):
    print("sim"+str(i)+" begin!")
    gen_data(i)