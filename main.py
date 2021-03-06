import sys
import numpy as np
from numpy.linalg import inv, det
from math import floor
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix,linalg
import gif

# Constants
defaultInputs = 6
tol = 1e-6
h = 1.0 # Elements size
E = 1000 # Young's Module
nu = 0.3 # Poisson ratio
t = 1.0 # Thickness
nodesElement = 4
dofsNode = 2
dofsElements = dofsNode*nodesElement
NuemannCase = 'Puntual Force'
DirichletCase = 'Clamp'

# Classes
class Mesh():
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
        self.nElms = nx*ny
        self.nNodes = (nx+1)*(ny+1)
        self.dofs = self.nNodes*dofsNode
        # Coordinate 
        x = np.linspace(0,h*nx,nx+1)
        y = np.linspace(0,h*ny,ny+1)
        x, y = np.meshgrid(x,y)
        self.xy = np.array([x.flatten(),y.flatten()])    
        # Topology
        n1 = np.array([iElm+floor(iElm/nx) for iElm in range(nx*ny)])
        n2 = n1 + 1
        n3 = n2 + nx + 1
        n4 = n1 + nx + 1
        self.Topology = np.array([n1,n2,n3,n4])    

class Material():
    def __init__(self):
        # Plain Stress
        self.D = (E/(1-nu**2))*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    
class TopOpt():
    def __init__(self,x,xold,v,r,p):
        self.x = x
        self.xold = xold
        self.v = v
        self.r = r
        self.p = p
        self.c = 0.0
        self.dc = np.zeros(x.shape)
        self.dv = np.ones(x.shape)
         
    def CreateFilter(self,Mesh):
        self.H = np.zeros((Mesh.nElms,Mesh.nElms))
        coords = Mesh.xy[:,Mesh.Topology.T.flatten()]
        x = np.sum(coords[0,:].reshape(Mesh.nElms,4),axis=1)/4
        y = np.sum(coords[1,:].reshape(Mesh.nElms,4),axis=1)/4
        center = np.array([x,y])
        for iElm in range(0,Mesh.nElms):
            for jElm in range(0,Mesh.nElms):
                d = np.sum(np.subtract(center[:,jElm],center[:,iElm])**2,axis=0)**0.5
                self.H[jElm,iElm] = np.maximum(0,self.r-d)
    
    def FilterSensitivities(self):
        sumH = np.sum(self.H,axis=0)
        self.dc = self.H@(self.x*self.dc)/(self.x*sumH[:,None])
    
    def OptimalityCriteria(self,Mesh):
        # Bisection Method
        l1 = np.amin(np.absolute(self.dc))/np.amax(np.absolute(self.dv))
        l2 = np.amax(np.absolute(self.dc))/np.amin(np.absolute(self.dv))
        move = 0.2
        while abs((l2-l1)/(l2+l1))>=tol:
            lmid = 0.5*(l1+l2)
            xnew = np.maximum(1e-3,np.maximum(self.x-move,np.minimum(1.0,np.minimum(self.x+move,self.x*np.sqrt(-self.dc/(self.dv*lmid))))))
            if np.sum(xnew) - self.v*Mesh.nElms > 0:
                l1 = lmid
            else:
                l2 = lmid
        self.x = xnew
        
class Plots():
    def Displacements(self,Mesh,Solution):
        # Plot Displacements
        x = np.arange(0,Mesh.dofs,2)
        y = np.arange(1,Mesh.dofs,2)
        ux = Solution[x]
        uy = Solution[y]
        d = (ux**2+uy**2)**0.5
        x = Mesh.xy[0,:].reshape(ny+1,nx+1)
        y = Mesh.xy[1,:].reshape(ny+1,nx+1)
        plt.pcolor(x,y,d.reshape(ny+1,nx+1))
        plt.colorbar()
        plt.show()
    
    @gif.frame
    def MaterialDistribution(self,Top,):
        # Plot material distribution
        # plt.clf()
        plt.pcolor(top.x.reshape(ny,nx),cmap='cool',edgecolors='k',linewidth=0.1)
        plt.colorbar()
        ax = plt.gca()
        ax.axis('equal')
        ax.axis('off')
        # plt.draw()
        # plt.pause(1e-6)
                     
class FEM():        
    def K(self,Mesh,Material,Top):
        # Integration
        # Initialize some variables
        xyGPI = np.array([[-0.5774,-0.5774,0.5774,0.5774],[-0.5774,0.5774,-0.5774,0.5774]])
        hGPI = np.array([1,1,1,1])
        refNodes = Mesh.Topology[:,0] # Pick a reference element (are the same)
        refVerts = Mesh.xy[:,refNodes]
        Ke = 0
        for iGP in range(4):
            Xi = xyGPI[0,iGP]
            Eta = xyGPI[1,iGP]
            H = hGPI[iGP]
            dNl = ShapeFunctions(Xi,Eta,1)
            Jacobian = refVerts@dNl.T
            dNg = inv(Jacobian)@dNl
            B = np.array([[dNg[0,0],0,dNg[0,1],0,dNg[0,2],0,dNg[0,3],0],[0,dNg[1,0],0,dNg[1,1],0,dNg[1,2],0,dNg[1,3]],[dNg[1,0],dNg[0,0],dNg[1,1],dNg[0,1],dNg[1,2],dNg[0,2],dNg[1,3],dNg[0,3]]])
            BtD = np.dot(B.T,Material.D)
            Ke = Ke + t*np.dot(BtD,B)*det(Jacobian)*H  
        # Assembly
        elsDofs = 2*np.kron(Mesh.Topology,np.ones((2,1)))+np.tile(np.array([[0],[1]]),[4,Mesh.nElms])
        row = np.kron(elsDofs,np.ones((1,dofsElements))).T.flatten()
        col = np.kron(elsDofs,np.ones((dofsElements,1))).T.flatten()
        K0 = np.tile(Ke.flatten(),(Mesh.nElms,1))
        data = np.repeat(Top.x,dofsElements**2,axis=1)**Top.p*K0
        K = csc_matrix((data.flatten(),(row,col)),shape=(Mesh.dofs,Mesh.dofs))
        return K,K0
    
    def Loads(self,Mesh):
        self.F = np.zeros((Mesh.dofs,1))
        if NuemannCase == 'Puntual Force':
            # Punctual Force
            maxValues = Mesh.xy.max(1)
            xMax = Mesh.xy[0,:]==maxValues[0]
            minValues = Mesh.xy.min(1)
            yMin = Mesh.xy[1,:]==minValues[1]
            nodes = np.arange(Mesh.nNodes)
            forceNodes = nodes[np.where(np.logical_and(xMax,yMin))]
            forceDofs = np.array([2*forceNodes,2*forceNodes+1])
            fx = 0
            fy = -1
            self.F[forceDofs[0,:]] = fx
            self.F[forceDofs[1,:]] = fy
            
    def Solver(self,Mesh,K):
        totalDofs = np.arange(Mesh.dofs)
        if DirichletCase == 'Clamp':
            minValues = Mesh.xy.min(1)
            xMin = Mesh.xy[0,:]==minValues[0]
            restNodes = np.where(xMin)[0]
            restDofs = np.array([2*restNodes,2*restNodes+1]).T.flatten()
            uRest = np.zeros((restDofs.size,1))
            freeDofs = np.setdiff1d(totalDofs,restDofs)
            
        freeDofsX,freeDofsY = np.meshgrid(freeDofs,freeDofs)
        frDofsX,frDofsY = np.meshgrid(restDofs,freeDofs)
        self.u = np.zeros((Mesh.dofs,1))
        uFree = linalg.inv(K[freeDofsX,freeDofsY])*self.F[freeDofs]-K[frDofsX,frDofsY]*uRest
        self.u[freeDofs] = uFree
        self.u[restDofs] = uRest
        return self.u

# Functions
def ShapeFunctions(Xi,Eta,dFlag):
    if dFlag==1:
        dN1dXi = -0.25*(1-Eta)
        dN1dEta = -0.25*(1-Xi)
        dN2dXi = 0.25*(1-Eta)
        dN2dEta = -0.25*(1+Xi)
        dN3dXi = 0.25*(1+Eta)
        dN3dEta = 0.25*(1+Xi)
        dN4dXi = -0.25*(1+Eta)
        dN4dEta = 0.25*(1-Xi)
        N = np.array([[dN1dXi,dN2dXi,dN3dXi,dN4dXi],[dN1dEta,dN2dEta,dN3dEta,dN4dEta]])
    else:
        N1 = 0.25*(1-Xi)*(1-Eta)
        N2 = 0.25*(1+Xi)*(1-Eta)
        N3 = 0.25*(1+Xi)*(1+Eta)
        N4 = 0.25*(1-Xi)*(1+Eta)
        N = np.array([N1,N2,N3,N4])       
    return N


if __name__ == '__main__':
    # User input
    if len(sys.argv) == defaultInputs:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        r = float(sys.argv[3])
        v = float(sys.argv[4])
        p = float(sys.argv[5])
    else:
        nx = 30
        ny = 20
        r = 1.5
        v = 0.3
        p = 3
    # Initialize FEM variables
    material = Material()
    mesh = Mesh(nx,ny)
    fem = FEM()
    # Initialize plot options
    plot = Plots()
    # plt.ion()
    # Initialize Optimization variables
    x = v*np.ones((nx*ny,1))
    xold = x
    top = TopOpt(x,xold,v,r,p)
    top.CreateFilter(mesh)
    change = 1000
    it = 0
    frames = []
    while change > 1e-3 and it < 1000:
        it += 1
        # Plot Material distribution
        frame = plot.MaterialDistribution(top)
        frames.append(frame)
        # Solve Elasticty Problem
        K,K0 = fem.K(mesh,material,top)
        fem.Loads(mesh)
        u = fem.Solver(mesh,K)
        # Evaluate Compliance and Sensitivities
        top.c = 0
        for iElm in range(0,mesh.nElms):
            iNodes = mesh.Topology[:,iElm]
            iDofs = np.array([2*iNodes,2*iNodes+1]).T.flatten()
            iK = K0[iElm,:].reshape(dofsElements,dofsElements)
            top.dc[iElm] = -top.p*top.x[iElm]**(top.p-1)*u[iDofs].T@iK@u[iDofs]
            top.c = top.c + top.x[iElm]**(top.p)*u[iDofs].T@iK@u[iDofs]
        # Update material distribution
        top.FilterSensitivities()
        top.OptimalityCriteria(mesh)
        change = np.amax(abs(top.x-top.xold))
        print('Iteration: {} | Compliance: {} | Change: {}'.format(it,top.c[0,0],change))
        top.xold = top.x
    gif.save(frames,'TopOpt.gif',duration=50)