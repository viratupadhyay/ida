# Python scripts to support batch plotting
import numpy as np
import matplotlib as plt

def plotc(data, wdir='.', name='c', time=0, lim=1):
    """Save a contour plot

    Usage: plotc(wdir, name='c', time=0, lim=1)
    Data files created from names: wdir/data/name.time
    Output (png) files: wdir/imgs/name_num-x.png
    Limits (contour plots) for h are [1,lim], qy [-lim,lim]; else [0,lim]
    """

    import ida

    if lim == 'None':
        limits=[None,None]
    else:
        limits=[0,lim]
        if name == 'c1' or name == 'c2':
            limits=[0,lim]
        if name == 'phi':
            limits=[0,lim]
        if name == 'vy':
            limits=[-lim,lim]

    fname = ida.fname(wdir, name, time)
    ida.cplot(data, limits, 'jet',fname[1])

    return

# Python script for axial (x) plots

def plotx(data, wdir='.', name='c', time=0):
    """Save an axial (x) plot of min, max average average across the channel

    Usage: plotx(wdir, name='c', time=0)
    Output (png) file wdir/plts/name_num-x.png
    """

    import ida
    fname = ida.fname(wdir, name, time)
    label = name + '(' + str(time) + ')'
    ida.xplot(data, limits=[0,1], fname=fname[2], label=label)

    return

# Python script for transverse (y) plots

def ploty(data, wdir='.', name='c', time=0, xval=[0.5]):
    """Save transverse (y) plots for different x (listed in xval)

    Usage: plotx(wdir, name='c', time=0, xval=[0.5])
    Output (png) file wdir/plts/name_num-y.eps
    """

    import ida

    fname = ida.fname(wdir, name, time)
    label = name + '(' + str(time) + ')'
    ida.yplot(data, fname=fname[2], xval=xval, label=label)

    return

# Python batch script for saving plots

def Bplot(inputs):
    """Batch contour plots

    Usage: python bplot 'wdir', 'times'
        Examples: wdir  = '/work/dissoltest'
                  times = '0,10,2'  (t=0, t=2, t=4, t=6, t=8)
    Makes a contour plot from wdir/name_time.dat.
    Output (png) file wdir/imgs/name_num-c.png
    Optional args (must be entered in order):
        'lim', 'names', 'types', 'func', 'xval'
    Limits for contour plots are: h [1,lim], qy [-lim,lim]; else [0,lim]
    """

    import os
    import ida
    import matplotlib as plt

#    Defaults
    names = ['c1']                            # concentration
    types = ['C']                            # contour plot
    lim   = None                            # limit on contours
    xval = '0.2,0.4,0.6,0.8'                # list for yplot

#    Mandatory arguments
    args  = len(inputs) - 1                    # Argument count
    if (args < 2):
        print "Usage: python bplot 'sim', 'times'"
        return

    sim  = inputs[1]
    times = inputs[2].split(',')
    times = range(int(times[0]), int(times[1])+1, int(times[2]))

#    Optional arguments
    if (args > 2): lim   = [float(x) for x in inputs[3].split(',')]
    if (args > 3): names = inputs[4].split(',')
    if (args > 4): types = inputs[5].split(',')
    if (args > 5): xval  = [float(x) for x in inputs[6].split(',')]

#    Make directories
    wdir = '/work/viratu/'+sim
    ldir = '/home/ladd/viratu/code/porous/'+sim
    os.system('mkdir -p '+wdir+'/imgs '+wdir+'/plts')
    #wdir = ldir
    #print(wdir)
#    Make plots
    n = 0
    Tip = []
    N = []
    for name in names:
        if (lim == None):
            clim = lim
        else:
            clim = lim[n]
            n = n + 1

        for time in times:
            print name, time
            #if ('N' in types):
            #    wdir = ldir
            if ('A' not in types):
                data = ida.paradata(wdir, name, time)
                Nm = ida.fname(wdir, name, time)
            #if time == 0:
             #   data = data+1
            if ('N' in types):
                Nact = ida.chactive(data,0.1,40)
                N = np.append(N,Nact)
                print Nact
            if ('D' in types):
                dc = ida.difference(wdir, name, time)
                nd = name + '-diff'
                plotc(dc, wdir, nd, time, clim)
            if ('V' in types):
                tipp = ida.tipcoord(data)
                Tip = np.append(Tip,tipp)
            if ('C' in types): plotc(data, wdir, name, time, clim)
            if ('X' in types): plotx(data, wdir, name, time)
            if ('Y' in types): ploty(data, wdir, name, time, xval)
            if ('A' in types): 
                dx = clim
                ida.plotxx(wdir, name, dx, time)
            if ('Z' in types):
                os.system('mkdir -p '+ldir+'/vtk')
                tm = np.str(time)
                znm = ldir+'/vtk/'+name + tm + '.vtk'
                ida.vtk_format(data,znm)
            #if ('N' not in types):
            #os.system('rm '+ Nm[0] + '*')    #Remove the files from work directory
            if ('D' in types):
                Nmd = ida.fname(wdir+'-seed', name, time)
                os.system('rm '+ Nmd[0] + '*')
        if ('V' in types):
            np.savetxt(wdir+"/Tip.dat",Tip)
            np.savetxt(ldir+"/Tip.dat",Tip)
        if ('N' in types):
            np.savetxt(ldir+"/Nact.dat",N)
    os.system('cp -r '+wdir+'/imgs '+ldir+'/')
    os.system('cp -r '+wdir+'/plts '+ldir+'/')
    return

if __name__ == '__main__':
    import sys
    Bplot(sys.argv)
