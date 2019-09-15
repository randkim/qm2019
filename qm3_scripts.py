import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fftpack import fft, ifft

#ONE_DIM_DVR DOES NOT WORK. DEBUG FIRST BEFORE USING
class one_dim_DVR:
    
    def __init__(self, left_end, right_end, no_of_points):
        #Trap and number of points considered
        self.left_end = left_end
        self.right_end = right_end
        self.no_of_points = no_of_points
        
        self.length = self.right_end - self.left_end
        self.no_of_useful_points = self.no_of_points - 1
        
        #Position reference
        x = []
        for i in range(self.no_of_points):
            dist = self.left_end + self.length * i/self.no_of_points
            x.append(np.round(dist,2))  
        self.x = x
        
        #For calculations
        n = []
        for i in range(self.no_of_useful_points):
            n.append(i)
        self.n = n
        
        nsquared = []
        for i in range(self.no_of_useful_points):
            nsquared.append(i**2)
        self.nsquared = nsquared
        
        #Empty arrays for kinetic/potential/Hamiltonian matrix
        self.kinetic = np.empty([self.no_of_useful_points, self.no_of_useful_points])
        self.potential = np.empty([self.no_of_useful_points, self.no_of_useful_points])
        self.hamiltonian = np.empty([self.no_of_useful_points, self.no_of_useful_points])
        
        #Kinetic term in DVR Basis
        self.constant_kinetic = (1/2)*(np.pi/self.length)**2 * (2/no_of_points)
        
        step = []
            
        for i in range(self.no_of_useful_points):
            for j in range(self.no_of_useful_points):
                for k in range(self.no_of_useful_points):
                    step.append(self.nsquared[k] * np.sin(self.n[k]*np.pi*i/self.no_of_points) * np.sin(self.n[k] * np.pi *j/self.no_of_points))
                self.kinetic[i,j] = np.sum(step)
                    
        self.kinetic = self.kinetic * self.constant_kinetic

        
    #External/Perturbing potential term. Add onto the list if you need any other functions of x to be accounted for
    def ext_potential(self, pre_factors, pot_eqn):
        self.pre_factors = float(pre_factors)
        self.pot_eqn = pot_eqn
           
        if pot_eqn == 'x':
            for i in range(self.no_of_useful_points):
                self.potential[i,i] = self.pre_factors * self.x[i]
            self.potential = np.diag(np.diag(self.potential))
            
        elif "x^" in pot_eqn:
            for i in range(self.no_of_useful_points):
                self.potential[i,i] = self.pre_factors * self.x[i]**int(pot_eqn[2]) 
            print('The potential equation is: ', pot_eqn)
            self.potential = np.diag(np.diag(self.potential))
            
        elif pot_eqn == "log(x)":
            log_x = []
            for i in range(self.no_of_useful_points):
                log_x.append(np.log(self.x[i]))
                self.potential[i,i] = self.pre_factors * self.log_x[i]
            self.potential = np.diag(np.diag(self.potential))
            
        else:
            print("Please implement this function manually")
            
        return self.potential
        
    
    def find_solution(self):
        self.hamiltonian = self.kinetic + self.potential
        
        self.eigenenergy, self.eigenvector = LA.eig(self.hamiltonian)
        
        return self.eigenenergy, self.eigenvector
    
    
    def plotter(self, plotrange, energy):
        plot_range = []
        for i in range(plotrange):
            plot_range.append(i)
            
        plt.bar(plot_range, energy[0: plotrange])
        plt.show()
        
        
        
#Need to implement time-dependent states/potential        
class split_operator_FFT():
    
    def __init__(self, left_end = -20, right_end = 20, no_of_points = 512, evolution_period = np.pi, no_of_time_steps = 10*10**(3), gaussian_fwhm = 5):
        #Trap and number of points considered
        self.left_end = left_end
        self.right_end = right_end
        self.length = self.right_end - self.left_end
        
        if math.log(no_of_points,2).is_integer():
            self.no_of_points = no_of_points
        else:
            print("Your number of points should be a power of 2. The program will now exit.")
            exit()
        
        #Position reference
        x = []
        for i in range(0, self.no_of_points-1):
            x.append(self.left_end + self.length*i/self.no_of_points) 
        self.x = x
        
        #Momentum reference
        #Negative values after halfway through to account for negative momentum
        p = []
        p_constant = 2*np.pi/self.length
        for i in range(0, int(self.no_of_points/2 - 1)):
            p.append(p_constant * i)
        for i in range(int(-self.no_of_points/2), 0):
            p.append(p_constant * i)
        self.p = p
        
        if len(self.x) - len(self.p) != 0:
            print("Position and Momentum arrays have different length")
        
        self.evolution_time = evolution_period
        self.no_of_time_steps = no_of_time_steps
        self.time_step = self.evolution_time/self.no_of_time_steps
        
        potential = []
        for i in range(0, len(self.x)):
            potential.append(np.exp(-1j * (0.5*self.x[i]**2) * self.time_step/2))
        self.potential = potential
        
        kinetic = []
        for i in range(0, len(self.p)):
            kinetic.append(np.exp(-1j * (0.5 * self.p[i]**2) * self.time_step))
        self.kinetic = kinetic
        
        self.psi0 = signal.gaussian(len(self.x), gaussian_fwhm)
        plt.plot(self.x, self.psi0)
        
    def ext_potential(self, magnitude, pot_eqn):
        self.magnitude = float(magnitude)
        self.pot_eqn = pot_eqn
           
        if pot_eqn == 'x':
            for i in range(0, len(self.x)):
                self.potential[i] = self.potential[i] * np.exp(-1j * (self.magnitude * self.x[i]) * self.time_step/2)
            
        elif "x^" in pot_eqn:
            for i in range(0, len(self.x)):
                self.potential[i] = self.potential[i] * np.exp(-1j * (self.magnitude * self.x[i]**float(pot_eqn[2])) * self.time_step/2) 
            
        elif pot_eqn == "log(x)":
            for i in range(0, len(self.x)):
                self.potential[i] = self.potential[i] * np.exp(-1j * (self.magnitude * np.log10(self.x[i])) * self.time_step/2) 
            
        else:
            print("Please implement this function manually")
            
        return self.potential
    
    def solve(self):
        for i in range(0, int(self.no_of_time_steps)):        
            psi1 = self.potential * self.psi0
            psi2 = fft(psi1)
            psi3 = self.kinetic * psi2
            psi4 = ifft(psi3)
            psi5 = self.potential * psi4
            self.psi0 = psi5
            
        self.psi = self.psi0

        plt.plot(self.x, np.abs(self.psi))
            
hi = split_operator_FFT()
hi.ext_potential(0.1, 'x^4')
hi.solve()