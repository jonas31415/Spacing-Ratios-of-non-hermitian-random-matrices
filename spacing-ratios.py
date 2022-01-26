import cmath
import numpy as np
import numpy.linalg as linalg
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from scipy.stats import kde

M = 10 #matrix size
samplesize=50 #number of matrices you want to compute

def makeGinibre(M):
    u=np.random.randn(M,M)+complex(0,1)*np.random.randn(M,M)
    u=u*np.sqrt(M)
    return u

def makeAId(M):
    u=np.random.randn(M,M)+complex(0,1)*np.random.randn(M,M)
    u=(u*u.transpose())/(2*np.sqrt(M))
    return u

def makeAIId(M):
    N=int(M/2)
    a=np.random.randn(N,N)+complex(0,1)*np.random.randn(N,N)
    d=a.transpose()
    v=np.random.randn(N,N)+complex(0,1)*np.random.randn(N,N)
    v=(v*v.transpose())/2
    b=np.triu(v, 1) - np.tril(v, -1)
    w=np.random.randn(N,N)+complex(0,1)*np.random.randn(N,N)
    w=(w*w.transpose())/2
    c=np.triu(w, 1) - np.tril(w, -1)
    
    u = complex(0,1)*np.ones((M,M))
    u[0:N,0:N]=a
    u[0:N,N:M]=b
    u[N:M,0:N]=c
    u[N:M,N:M]=d
    u=u*1/np.sqrt(M)
    return u

def plotSpacingRatios(re, im, ensemble):
    nbins = 100 # 100 x 100 punkte
    fig = plt.figure()
    ax = fig.add_subplot(111)
    k = kde.gaussian_kde([re, im])  #kernel-density estimate
    xi, yi = np.mgrid[-1:1:nbins*1j, -1:1:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    B = zi.reshape(xi.shape)
    plot = ax.pcolormesh(xi, yi, B, shading='auto')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.colorbar(plot, ax=ax)
    plt.title("complex spacingratios for M=%i, "%M +ensemble)
    plt.xlabel('Re(s)')
    plt.ylabel('Im(s)')
    plt.show()

def plotComponents(sGinibre, sAId, sAIId):
    abs_gi=[]
    theta_gi=[]
    abs_AId=[]
    theta_AId=[]
    abs_AIId=[]
    theta_AIId=[]

    for i in range(0, M*samplesize):
        abs_gi.append(cmath.polar(sGinibre[i])[0])
        theta_gi.append(cmath.polar(sGinibre[i])[1])
        abs_AId.append(cmath.polar(sAId[i])[0])
        theta_AId.append(cmath.polar(sAId[i])[1])
        abs_AIId.append(cmath.polar(sAId[i])[0])
        theta_AIId.append(cmath.polar(sAId[i])[1])
        
    plt.hist(abs_gi, bins=110,density=True, alpha = 0.5,histtype=u'step', label='Ginibre',stacked=True)
    plt.hist(abs_AId, bins=110,density=True, alpha = 0.5,histtype=u'step', label='AI$^{\N{DAGGER}}$',stacked=True)
    plt.hist(abs_AIId, bins=110,density=True, alpha = 0.5,histtype=u'step', label='AII$^{\N{DAGGER}}$',stacked=True)
    plt.legend(loc='upper left')
    plt.xlabel('r')
    plt.ylabel('p(r)')
    plt.show()

    plt.hist(theta_AId, bins=100,density=True, alpha = 0.5, histtype=u'step', label='AI$^{\N{DAGGER}}$',stacked=True)
    plt.hist(theta_AIId, bins=100,density=True, alpha = 0.5,histtype=u'step', label='AII$^{\N{DAGGER}}$',stacked=True)
    plt.hist(theta_gi, bins=100,density=True, alpha = 0.5, histtype=u'step',label='Ginibre',stacked=True)
    plt.legend(loc='lower left')
    plt.xlabel('\u0398')
    plt.ylabel('P(\u0398)')
    plt.show()

def computeSpacingRatios(M, samplesize, makeFunction):
    rea=[] #real and imaginary part of eigenvalues
    ima=[]
    s=[] #spacing-ratios
    ew=[] #eigenvalues
    betrag=[]
    theta=[]
    abs_ew=[] #a

    for k in range(0, samplesize):
    
        u=makeFunction(M)

        ew=linalg.eigvals(u)
        for i in range(0,M):
            abs_ew.append(abs(ew[i]))

        for i in range(0, M):
            rea.append(ew[i].real)
            ima.append(ew[i].imag)

        tree=KDTree(np.c_[rea, ima])
        if makeFunction==makeAIId:
            for i in range(0, M):
                nn,index=tree.query([rea[i],ima[i]], k=5)
                s.append(((rea[index[2]]+complex(0,1)*ima[index[2]])-(rea[index[0]]+complex(0,1)*ima[index[0]]))/((rea[index[4]]+complex(0,1)*ima[index[4]])-(rea[index[0]]+complex(0,1)*ima[index[0]])))
        else:
            for i in range(0, M):
                nn,index=tree.query([rea[i],ima[i]], k=3)
                s.append(((rea[index[1]]+complex(0,1)*ima[index[1]])-(rea[index[0]]+complex(0,1)*ima[index[0]]))/((rea[index[2]]+complex(0,1)*ima[index[2]])-(rea[index[0]]+complex(0,1)*ima[index[0]])))
    
    return s

sGinibre=computeSpacingRatios(M, samplesize, makeGinibre)
plotSpacingRatios(np.real(sGinibre),np.imag(sGinibre), "Ginibre-Ensemble")

sAId=computeSpacingRatios(M, samplesize, makeAId)
plotSpacingRatios(np.real(sAId),np.imag(sAId), "AI$^{\N{DAGGER}}$-Ensemble")

sAIId=computeSpacingRatios(M, samplesize, makeAIId)
plotSpacingRatios(np.real(sAIId),np.imag(sAIId), "AII$^{\N{DAGGER}}$-Ensemble")

plotComponents(sGinibre, sAId, sAIId)