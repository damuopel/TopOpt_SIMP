import sys
import numpy as np
from numpy.linalg import inv, det
from math import floor
import matplotlib.pyplot as plt

# Constants
h = 1 # Elements size
E = 2.1e11 # Young's Module
nu = 0.3 # Poisson ratio
t = 1 # Thickness

# Classes
class Mesh():
	def __init__(self,nx,ny):
		self.nx = nx
		self.ny = ny
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

class BoundaryConditions():
	def __init__(self):
		pass

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
	nx = sys.arg[1]
	ny = sys.arg[2]
	r = sys.arg[3]
	v = sys.arg[4]
	# Initialize variables
	x = v*np.ones(nx*ny)
	material = Material()
	change = 1000
	while change > 1e-4:
		pass



