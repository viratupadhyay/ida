# Interactive Data Analysis Module

import numpy as np, matplotlib.pyplot as plt, ida, math, os
#from skimage import measure

def fname(wdir='.', type='c', time=0):
    """Create file names

    Usage: fname = fname(wdir='.', type='c', time=0)
    Returns file names wdir/data(or imgs)/name.time
    """

    dtime = "%.4d" % time
    datname = wdir + '/data/' + type + '.' + dtime
    imgname = wdir + '/imgs/' + type + '.' + dtime
    pltname = wdir + '/plts/' + type + '.' + dtime
    return [datname, imgname, pltname]


def fname_init(ldir='.', type='phi'):
    name = ldir + '/init/' + type + '.dat'
    return name

def fread(fname):
    """Read a data file
    
    Usage: data = fread(fname)
    Returns a float array of dimensions Nx x Ny
    """

    fhandler = open(fname)
    header = fhandler.readline().split()
    data = fhandler.read().split()
    Nx = int(header[2])
    Ny = int(header[3])
    Nz = int(header[4])
    fdata = np.array(data, dtype='f')
    if (Nz == 1):
        fdata.shape = (Nx,Ny)
    else:
        fdata.shape = (Nx,Ny,Nz)
    return fdata

def fwrite(data, fname):
    f = open(fname, 'w')
    nx, ny = data.shape
    Nx = nx*8
    Ny = ny*4
    header = '%5d'%Nx+'%5d'%Ny+'%5d'%nx+'%5d'%ny+'%5d'%1+'    0'+'    0.0'
    f.write(header)
    f.write('\n')
    for m in range(nx):
        for n in range(ny):
            l = ' % .5e'%data[m][n]
            f.write(l)
            if ((m*ny+n+1)%10 == 0):
                f.write('\n')
    f.close();


def pget(workdir = '.', type='c', time=0):
    "name = simulation name"
    dtime = "%.4d" % time
    if (os.path.isfile(workdir+'/data/'+type+'.'+dtime+'.00') == True):
        return
    else:
        hosts = np.loadtxt(workdir+'/init/hfile.txt',dtype='S3')
        for i in range(np.size(hosts)):
            rank = "%.2d" % i
            os.system("scp "+hosts[i]+":"+workdir+'/data/'+type+'.'+dtime+'.'+rank+' '+workdir+'/data/')
    
    
def paradata(wdir='.', type = 'phi', time = 0):
    " Read array with rank = 0 and read Np and Nq from header file "
    name = fname(wdir, type, time)
    name0 = name[0] + '.' + "%.2d" % 0
    pget(wdir,type,time)
    fhandler = open(name0)
    header = fhandler.readline().split()
    Np = int(header[5])
    Nq = int(header[6])
    size = Np*Nq
    for rank in range(size+1):
        if (rank < size):
            name_rank = name[0] + '.' + "%.2d" % rank
            print name_rank
            data = fread(name_rank)
        if (rank % Nq == 0):
            if (rank == Nq):
                datarr = datacol
                datacol = data
            elif (rank > 0):
                datarr = np.concatenate((datarr,datacol),axis=0)
                datacol = data
            else:
                datacol = data
        else:
            datacol = np.concatenate((datacol,data),axis=1)
    return datarr


def paradata_init(ldir='.', type = 'phi', dim=2):
    name = fname_init(ldir,type)
    name0 = name + '.' + "%.2d" % 0
    fhandler = open(name0)
    header = fhandler.readline().split()
    Np = int(header[0])/int(header[2])
    Nq = int(header[1])/int(header[3])
    size = Np*Nq
    for rank in range(size+1):
        if (rank < size):
            name_rank = name + '.' + "%.2d" % rank
            data = fread(name_rank)
        if (rank % Nq == 0):
            if (rank == Nq):
                datarr = datacol
                datacol = data
            elif (rank > 0):
                datarr = np.concatenate((datarr,datacol),axis=0)
                datacol = data
            else:
                datacol = data
        else:
            datacol = np.concatenate((datacol,data),axis=1)
    if (dim == 2):
        datarr = datarr.reshape(int(header[0]),int(header[1]))

    return datarr


def seed(ldir = '.', type = 'phi', A = 0.1, alp = 0.01, b = 0.001):
    name = fname_init(ldir,type)
    name0 = name + '.' + "%.2d" % 0
    fhandler = open(name0)
    header = fhandler.readline().split()
    hinit = ida.paradata_init(ldir,type,2)
    Nx = hinit.shape[0]
    Ny = hinit.shape[1]
    nx = int(header[2])
    ny = int(header[3])
    Np = Nx/nx
    Nq = Ny/ny
    for x in np.arange(Nx):
        Ai = A*np.exp(-alp*x)
        if (Ai > 1e-6):
            for y in np.arange(Ny):
    #            hinit[x][y] += Ai*(np.sin(2*n*np.pi*y/Ny))
                hinit[x][y] += Ai*np.exp(-b*((y-Ny/2)**2))
    for i in range(Np):
        for j in range(Nq):
            rank = i*Nq+j
            data = hinit[i*nx:(i+1)*nx]
            data = np.transpose(data)
            data = data[j*ny:(j+1)*ny]
            data = np.transpose(data)
            name_seed = name+'-seed.'+"%.2d" % rank
            #name_seed = ldir+'/seed'+num+'/'+type+'.dat'+'.'+"%.2d" % rank
            f = open(name_seed, 'w')
            header = '%5d'%Nx+'%5d'%Ny+'%5d'%nx+'%5d'%ny+'%5d'%1+'    0'+'    0.0'
            f.write(header)
            f.write('\n')
            for m in range(nx):
                for n in range(ny):
                    l = ' % .5e'%data[m][n]
                    f.write(l)
                    if ((m*ny+n+1)%10 == 0):
                        f.write('\n')
            f.close();
    sd = ldir+'/seed.dat'
    f = open(sd, 'w')
    f.write('A      alp     b\n')
    l = '% 2.2f'%A+'% 2.2f'%alp+'% 2.4f'%b
    f.write(l)
    f.close()
    return hinit



def vtk_format(data, name='data.vtk'):
    nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]
    f = open(name, 'wb')
    f.write("# vtk DataFile Version\n")
    f.write("3D visualization of data\n")
    f.write("ASCII\n\n")
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write("DIMENSIONS %1d %1d %1d\n" % (nx, ny, nz))
    f.write("ORIGIN 0 0 0\n")
    f.write("SPACING 1.00000 1.00000 1.00000\n\n")
    f.write("POINT_DATA %1d\n" % (nx*ny*nz))
    f.write("SCALARS Data float\n")
    f.write("LOOKUP_TABLE default\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                line = []
                line = np.str(data[i][j][k])
                f.write(line)
                f.write("\n")
    f.close()


def cplot(data, limits=[None,None], CM = 'jet', fname='', ext='png'):
    """Make a color contour plot of data

    Usage: cplot(data, limits=[None,None], fname='')
    If no filename is specified a plot is displayed
    File format is ext (default is png)
    """

    SIZE = 12
    DPI  = 100

    nx, ny = data.shape[0], data.shape[1]
    data = data.reshape(nx,ny)
    scale  = SIZE/float(max(nx,ny))
    plt.figure(figsize=(scale*nx, scale*ny+1.0))
    plt.clf()
    c = plt.imshow(np.transpose(data), cmap=CM)
    plt.clim(limits)
    plt.axis([0,nx,0,ny])
    #cbar = plt.colorbar(c, ticks=np.arange(0.831,0.835,0.001), aspect = 20, orientation='vertical', shrink=0.72, extend='neither', spacing='proportional')
    #cbar = plt.colorbar(c, aspect = 40, orientation='vertical', shrink=0.72, extend='neither', spacing='proportional')
    #cbar = plt.colorbar(c, orientation='horizontal', shrink=1.0)
    cbar = plt.colorbar(c, orientation='vertical', shrink=0.72, extend='neither', spacing='proportional')
    
    cbar.ax.tick_params(labelsize=21,size=10)
    #cbar.ax.yaxis.set_ticks_position('left')
    #c.cmap.set_under(color='black')
    if len(fname) == 0:
        plt.show()
    else:
        plt.savefig(fname+'.'+ ext, format=ext, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def xplot(data, limits=[None,None], fname='', func='max', label='',
        loc='upper right', ext='png'):
    """Make an axial plot of a funtion of data

    Usage: xplot(data, limits=[None,None], fname='', loc='upper left', ext='png')
    Possible functions to plot are max, min, avg
    If no filename is specified a plot is displayed
    The special filename 'M' turns "hold" on (for multiplots)
    File format is ext (default is png)
    """

    nx,ny = data.shape[0],data.shape[1]
    z = np.zeros([3,nx])
    for x in range(nx):
        z[0,x] = data[x,:].min()
        z[1,x] = data[x,:].max()
        z[2,x] = data[x,:].sum()/float(ny)

    #plt.plot(z[0], label=label+' min')
    #plt.plot(z[1], label=label+' max')
    plt.plot(z[2], label=label+' avg')

    if (fname == 'M'):
        plt.hold=True
    else:
        plt.legend(loc=loc)
        plt.axis([0,data.shape[0]-1,limits[0],limits[1]])
        plt.hold=False
        if len(fname) == 0:
            plt.show()
        else:
            plt.savefig(fname+'-x.'+ext, format=ext)
            plt.close()


def yplot(data, limits=[None,None], fname='', xval=[0.5], label='',
        loc='upper left', ext='png'):
    """Make transverse plots of data

    Usage: yplot(data, limits=[None,None], fname='', xval=[0.5], label='',
        loc='upper left', ext='ext')
    xval is a list of axial distances in units of nx
    If no filename is specified a plot is displayed
    The special filename 'M' turns "hold" on (for multiplots)
    File format is ext (default is png)
    """

    nx, ny = data.shape[0], data.shape[1]

    y = np.array(range(ny)) + 0.5
    for x in xval:
        ix = int(x*nx)
        plt.plot(y, data[ix,:], label=label+" : "+"x="+str(x))

    if (fname == 'M'):
        plt.hold=True
    else:
        plt.axis([0,data.shape[1],limits[0],limits[1]])
        plt.legend(loc=loc)
        plt.hold=False
        if len(fname) == 0:
            plt.show()
        else:
            plt.savefig(fname+'-y.'+ext, format=ext)
            plt.close()


def chtrack(h, hcrit=2.0, xlim=40):
    nx, ny = h.shape[0], h.shape[1]
    c = measure.find_contours(h, hcrit)
    n = np.shape(c)[0]
    X = []
    Y = []
    for i in range(n):
        cx0 = c[i][0,0]
        cy0 = c[i][0,1]
        cx1 = c[i][(np.shape(c[i])[0]-1),0]
        cy1 = c[i][(np.shape(c[i])[0]-1),1]
        if (cx0 != cx1):
            if (cy0 != cy1):
                xi = c[i][:,0]
                yi = c[i][:,1]
                xi = xi.tolist()
                yi = yi.tolist()
                X = X + xi              #Appending the list
                Y = Y + yi              #Appending the list
    X = np.array(X)
    Y = np.array(Y)
    X1 = np.roll(X,1)
    nch = np.zeros(int(max(X)))
    for k in range(xlim):
        I = np.logical_xor(X <= k, X1 > k)
        iI = [i for i, j in enumerate(I) if j == False and i != 0]
        nch[k] = np.size(iI)/2
    tx = np.zeros(int(nch[xlim-1]))
    ty = []
    if X[iI[0]] > (xlim-1):             # If 1st peak is within the fracture
        for i in range(int(nch[xlim-1])-1):
            peak = X[iI[2*i]:iI[(2*i+1)]]
            tx[i] = max(peak)
            ty = ty + [Y[(j+iI[2*i])] for j,k in enumerate(peak) if k == max(peak)]
    else:                               # If 1st peak is periodic at the end
        peak0 = X[0:iI[0]]
        tx[0] = max(peak0)
        ty = [Y[j] for j,k in enumerate(peak0) if k == max(peak0)]
        for i in range(int(nch[xlim-1])-1):
            peak = X[iI[2*i+1]:iI[2*(i+1)]]
            tx[i+1] = max(peak)
            ty = ty + [Y[(j+iI[2*i])] for j,k in enumerate(peak) if k == max(peak)]
    for i in range(xlim,int(max(X))):
        tips = np.logical_not(tx[0:len(tx)] < i)
        tips = tips.tolist()
        nch[i] = tips.count(True)
    ty = np.array(ty)
    return nch,X,Y,tx,ty


def chactive(c, ccrit = 0.005, xlim = 10):
    nx, ny = c.shape[0], c.shape[1]
    ch = c[xlim]
    nch = np.size(ch)
    N = 0
    for i in range(1,nch-1):
        if ch[i-1]<ch[i] and ch[i+1]<ch[i]:
            if ch[i] > ccrit:
                N = N+1
    return N


def Nch_write(nch, fname):
    N = np.size(nch)
    x = np.arange(N)
    A = [x,nch]
    A = np.array(A)
    A = np.transpose(A)
    np.savetxt(fname, A, fmt="%4d")


def Nch_logscale(fname):
    A = np.transpose(np.genfromtxt(fname))  # Read N(L) for each x 
    x = A[0]                        # L values
    N = A[1]                        # N values
    X = np.arange(1,3,0.02)         # x = 10-1000 with equidistant points in log-scale
    Xe = np.ceil(np.power(10,X))    # Find the nearest integer
    n = np.size(Xe)
    Ne = np.zeros(n)                # N(L) at that value of x
    for i in range(n):
        Ne[i] = N[Xe[i]]
    return Xe,Ne


def avgdata(data):
    nx, ny = data.shape[0], data.shape[1]
    avg = np.zeros(nx)
    Max = np.zeros(nx)
    for i in range(nx):
        avg[i] = sum(data[i,:])/ny
        Max[i] = max(data[i,:])
    return avg,Max


def lpdata(h):
    nx,ny = h.shape[0], h.shape[1]
    tx,ty = ida.tipcoord(h)
    ch=[]
    lp=0
    avgd = np.zeros(tx)
    for x in range(tx):
        vec = h[x,:]
        ch.append([x for i in vec if i>2.0])
        if(np.size(ch[x])==ny):
            lp=x
    return lp,tx,ty


def chline(h, hcrit=2.0):
    nx,ny = h.shape[0], h.shape[1]
    tx,ty = ida.tipcoord(h)
    ch = np.zeros(nx*ny)
    ch.shape = (nx,ny)
    for x in range(tx):
        for y in range(ny):
            if(h[x][y])<hcrit:
                ch[x][y]=0
            else:
                ch[x][y]=1
    return ch


def tipcoord(hdata):
    """Finds the coordinate of tip of the longest channel based on the aperture data
     
    Usage: tipcoord(h_data)
    """ 

    nx, ny = hdata.shape[0], hdata.shape[1]
    z = np.zeros(nx)
    tx=0
    for x in range(nx):
        z[x] = hdata[x,:].max()
        if z[x] > 2.0:
            tx = x
        else:
            break
    for y in range(ny):
        if hdata[tx][y] == hdata[tx].max():
            ty = y
    return tx,ty


def cline(qx,qy,h):
    """Finds the center-line flow in the channel, i.e. the path of the maximum flow rate in a     channel. It uses quadratic interpolation to find the exact location where the flow rate is max    imum between two pixels.

    Usage: cline(qx_data, qy_data, h_data)
    """
    tx,ty = ida.tipcoord(h)
    print tx,ty
    nx, ny = qx.shape[0], qx.shape[1]
    Q = np.sqrt(np.matrix(qx**2.0 + qy**2.0))
    Qmax = np.zeros(tx)
    ymax = np.zeros(tx)
    ymax2 = np.zeros(tx)
    for x in range(tx):
        Qmax[x] = Q[x,:].max()
        for y in range(ny):
            if Q[x,y] == Qmax[x]:
                ymax[x] = y
        A = np.matrix([[(ymax[x]-1)**2,ymax[x]-1,1],[(ymax[x])**2,ymax[x],1],[(ymax[x]+1)**2,ymax[x]+1,1]])
        B = np.matrix([[(Q[x,(ymax[x]-1)])],[(Q[x,(ymax[x])])],[(Q[x,(ymax[x]+1)])]])
        X = np.linalg.solve(A,B)
        ymax2[x] = (-X[1]/(2*X[0]))
    plt.plot(ymax2,Qmax)
    #plt.axis([0,h.shape[0],ymax2[0]-5,ymax2[0]+5])
    plt.show()
    return ymax2


def interp(data, h):
    """Quadratic Interpolation of data
    """
    nx, ny = data.shape[0], data.shape[1]
    #data = np.matrix(data)
    tx,ty = ida.tipcoord(h)
    ymax = np.zeros(tx)
    ymax2 = np.zeros(tx)
    datam = np.zeros(tx)
    for x in range(tx):
        datam[x] = data[x,:].max()
        for y in range(ny):
            if data[x,y] == datam[x]:
                ymax[x] = y
        A = np.matrix([[(ymax[x]-1)**2,ymax[x]-1,1],[(ymax[x])**2,ymax[x],1],[(ymax[x]+1)**2,ymax[x]+1,1]])
        B = np.matrix([[(data[x,(ymax[x]-1)])],[(data[x,(ymax[x])])],[(data[x,(ymax[x]+1)])]])
        X = np.linalg.solve(A,B)
        ymax2[x] = (-X[1]/(2*X[0]))
    #plt.plot(ymax2)
    #plt.axis([0,nx,0,ny])
    #plt.show()
    return ymax2


def avgQ(qx,qy,h):
	nx,ny = h.shape[0], h.shape[1]
	tx,ty = ida.tipcoord(h)
	ch=[]
	lp=0
	Q = np.zeros(tx)
	for x in range(tx):
		vec = h[x,:]
		ch.append([x for i in vec if i>2.0])
		if(np.size(ch[x])==ny):
			lp=x
	#ch = np.array(ch)
	for x in range(tx):
		for y in range(ny):
			if x <= lp:
				Q[x] += (math.sqrt((qx[x][y]**2.0+qy[x][y]**2.0)))/ny
				#Q[x] += p[x][y]/ny
			else:
				break
	for x in range(lp+1,tx):
		for y in range(ty-200,ty+200):
				if h[x][y] > 2.0:
					Q[x] += (math.sqrt((qx[x][y]**2.0+qy[x][y]**2.0)))#/np.size(ch[x])
				#	Q[x] += p[x][y]/np.size(ch[x])
	#plt.plot(Q)
	#plt.show()
	return Q


def difference(wdir=".", type = "c", time = 0):
    cn = ida.paradata(wdir,type,time)
    ws = wdir+'-seed'
    cs = ida.paradata(ws,type,time)
    dc = cs - cn
    return dc


def length(data):
    l = 0
    for i in range(np.size(data)-1):
        l += ((data[i+1]-data[i])**2+1)**0.5
    return l


def qch(qx,qy,ty,lp,chd):
    q = 0
    for y in range(ty-chd,ty+chd):
        q += np.sqrt(qx[lp+94][y]**2+qy[lp+94][y]**2)
    return q


def Qt(simname):
    A = np.genfromtxt("../PCON/"+simname+"/data/avg.csv")
    A = np.transpose(A)
    qavg = A[3][1:np.size(A[3])]
    t = A[1][1:np.size(A[1])]
    return t,qavg/1024


def plotxx (wdir='.', type='ck1', dx = 0.2, time = 0):
    f1 = ida.paradata(wdir,'f1',time)
    f2 = ida.paradata(wdir,'f2',time)
    c2 = ida.paradata(wdir,'c2',time)
    c1 = ida.paradata(wdir,'c1',time)
    nx,ny = c1.shape[0],c1.shape[1]
    f1x = np.zeros(nx)
    f2x = np.zeros(nx)
    c1x = np.zeros(nx)
    c2x = np.zeros(nx)
    X = np.zeros(nx)
    for x in range(nx):
        f1x[x] = f1[x,:].sum()/float(ny)
        f2x[x] = f2[x,:].sum()/float(ny)
        c1x[x] = c1[x,:].sum()/float(ny)
        c2x[x] = c2[x,:].sum()/float(ny)
        X[x] = x*dx
    cx = c1x+c2x
    dcx = np.diff(cx)
    flux = c1x[0:nx-1]+c2x[0:nx-1]-dcx/dx
    plt.plot(X,c1x,label=r'Coupling ion $X$ concentration $c_X$',color='red')
    plt.hold=True
    plt.plot(X,c2x,label=r'Secondary ion $A_S$ concentration $c_S$',color='green')
    plt.hold=True
    plt.plot(X,f2x,label=r'Primary specific volume $f_P$',color='blue')
    plt.hold=True
    plt.plot(X,f1x,'o',markersize=3, markeredgecolor='#9502f9', label=r'Secondary specific volume $f_S$',color='#9502f9')
    #plt.hold=True
    #plt.plot(X,cx,'o',color='black',markersize=3,markeredgecolor='black',label=r'$(c_X+c_S)$')
    #plt.hold=True
    #plt.plot(X[0:nx-1],flux,'o',color='m',markersize=3,markeredgecolor='m',label=r'$c_S+c_X-Pe^{-1}d(c_S+c_X)/dx$')
    plt.axis([0,20,-0.1,2])
    plt.legend(loc='upper right',prop={'size':10})
    plt.grid(True)
    fn = ida.fname(wdir,type,time)
    plt.savefig(fn[2]+'-x.png',format='png')
    plt.close()
    #plt.show()


def finit (dir='test', dx=0.1):
    nx = 128*(0.2/dx)
    ny = 64
    lx = 10*(0.2/dx)
    f1 = np.zeros(nx*ny)
    f2 = np.zeros(nx*ny)
    f1 = f1.reshape(nx,ny)
    f2 = f2.reshape(nx,ny)
    f1[0:lx] = 0.99
    f2[lx:nx] = 0.99
    f1 = f1.reshape(nx,ny,1)
    f2 = f2.reshape(nx,ny,1)
    f1name = '../'+dir+'/init/f1.dat.00'
    f2name = '../'+dir+'/init/f2.dat.00'
    ida.fwrite(f1,f1name)
    ida.fwrite(f2,f2name)
