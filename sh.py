from typing import Dict, Any, List, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh

from modelbase import ElectronicModel_ as ElectronicT

class IniconGenNormal(object):
    '''Generate different initial condition including x0 and p0'''
    def __init__(self, x0: ArrayLike, p0: ArrayLike, sigma: Optional[ArrayLike] = None) -> None:
        self.x0_ = np.array(x0, dtype=np.float64)
        self.p0_ = np.array(p0, dtype=np.float64)
        self.sigma_x = (sigma == None) and 10 / self.p0_ or sigma
        self.sigma_p = 0.5 / self.sigma_x
        self.rng = np.random.default_rng()
        
    def __call__(self) -> tuple[ArrayLike, ArrayLike]:
        x0 = self.rng.normal(self.x0_, self.sigma_x)
        p0 = self.rng.normal(self.p0_, self.sigma_p)
        return x0, p0

class TrajectorySurfaceHopping(object):
    """Class to propagate a single FSSH trajectory"""

    def __init__(self, 
                 model: ElectronicT, 
                 x0: ArrayLike,
                 k0: ArrayLike,
                 initial_state: ArrayLike,
                 **options: Any) -> None:
        # restart
        self.restart = False

        # initial condition
        self.model = model
        self.mass = model.mass
        self.nstate = model.nstate()
        self.kBT = np.linalg.norm(k0)**2 / (self.nstate * self.mass)
    
        self.position = x0
        self.last_velocity: ArrayLike
        self.velocity = k0 / self.mass
        self.wfc = np.array(initial_state, dtype=np.complex128)
        self.electronics = model.clone()
        self.electronics.compute(x0)

        # electronic step
        self.time: float = 0.0
        self.timestep: int = 0
        self.dt: float = options.get('dt', 0.02)
        self.electronic_integration: str = options.get('electronic_integration', 'exp')


        # TSH
        self.active: int = options.get('active', 0)
        self.hopping: bool = False
        self.tsh_mc: bool = options.get('MC', False)


        # boundary condition
        boundary: Dict = {}
        bounds = options.get('bounds', None)
        if bounds is not None:
            boundary['box_bound'] = np.array(bounds, dtype=np.float64)
        else:
            boundary['box_bound'] = None
        boundary['max_time'] = options.get('max_time', 10000.0)
        self.boundary = boundary
        

        # logger
        self.tracer = []
        self.snapevery = options.get('snapevery', 1)

        self.rng = np.random.default_rng()

    def snapshot(self) -> None:
        out = {
            'time': self.time,
            'position': self.position,
            'velocity': self.velocity,
            'active': self.active,
            'hopping': self.hopping
        }
        # self.tracer.append(out)
        self.tracer = [out]

    def continue_simulating(self) -> bool:
        '''whether need to continue simulating'''
        if self.time > self.boundary['max_time']:
            return False
        elif (self.boundary['box_bound'] is not None and
              np.any(self.position < self.boundary['box_bound'][0]) or
              np.any(self.position > self.boundary['box_bound'][1])):
            return False
        return True

    def nuclear_movement(self,
                         this_electronics: ElectronicT) -> tuple[ElectronicT, ElectronicT]:
        '''MD for ions with velocity verlet algorithm (Swope, 1982)
        '''
        dt = self.dt
        accerlation = this_electronics.force(state=self.active) / self.mass
        self.position += self.velocity * dt + 0.5 * accerlation * dt**2
        
        last_electronics = this_electronics
        this_electronics = this_electronics.update(self.position)

        next_accerlation = this_electronics.force(state=self.active) / self.mass
        self.last_velocity = self.velocity
        self.velocity += 0.5 * dt * (accerlation + next_accerlation)

        return last_electronics, this_electronics

    def propagate_electronics(self, 
                              last_electronics: ElectronicT,
                              this_electronics: ElectronicT,
                              ) -> ArrayLike:
        '''
        :return: electronic hamiltonian, n x n array
        '''
        # construst the hamiltonian in the midpoint
        H = 0.5 * (this_electronics.hamiltonian() + last_electronics.hamiltonian())
        NAC = 0.25 * np.einsum('ijx,x->ij', 
                               this_electronics._derivative_coupling + last_electronics._derivative_coupling,
                               self.velocity + self.last_velocity)
        H = H - 1j * NAC
        eigen_energy, eigen_wfc = eigh(H)

        # propagation
        propagator = np.diagflat(np.exp(-1j * eigen_energy * self.dt))
        self.wfc = eigen_wfc @ propagator @ eigen_wfc.conj().T @ self.wfc

        return H

    def surface_hopping(self, 
                        last_electronics: ElectronicT,
                        this_electronics: ElectronicT,
                        hamiltonian: ArrayLike):
        gijdt = -2.0 * self.dt * np.imag(hamiltonian[self.active,:] * self.wfc[self.active].conj() * self.wfc[:] ) \
                / np.abs(self.wfc[self.active])**2
        
        # zero out 'self-hop' for good measure (numerical safety)
        gijdt[self.active] = 0.0
        
        gijdt = np.maximum(gijdt, 0.0)

        # if self.tsh_mc:
        #     dE = 0.5 * (this_electronics.hamiltonian().diagonal() + last_electronics.hamiltonian().diagonal())
        #     gijdt = np.where(dE > 0,
        #                      gijdt * np.exp(-dE / self.kBT))

        rnd = self.rng.random()
        cumprob = np.cumsum(gijdt)
        hop_to = -1
        for i, p in enumerate(cumprob):
            if rnd < p:
                hop_to = i
                break

        if hop_to < 0:
            self.hopping = False
            return
        if not self.tsh_mc:
            hop = self._velocity_rescaling(this_electronics, hop_to)
            if not hop:
                self.hopping = False
                return

        self.hopping = True
        self.active = hop_to

    def _velocity_rescaling(self, 
                            electronic: ElectronicT, 
                            hop_to: int) -> bool:
        ''' Decide hop or not based on energy perservation. 
        Update the velocity.
        :return: True / False: frastrated hopping
        '''
        dij = self.electronics.derivative_coupling(self.active, hop_to)
        A = np.linalg.norm(dij)**2 / (2 * self.mass)
        B = np.vdot(dij, self.velocity)
        Delta = B**2 - 4 * A * (electronic.hamiltonian()[self.active, self.active] - electronic.hamiltonian()[hop_to, hop_to])
        if Delta < 0:
            return False
        sigma_m = (B - np.sqrt(Delta)) / (2 * A)
        sigma_p = (B + np.sqrt(Delta)) / (2 * A)
        sigma = (abs(sigma_m) < abs(sigma_p)) and sigma_m or sigma_p
        self.velocity -= dij * sigma / self.mass
        return True

    def decoherence(self):
        pass

    def compute(self) -> List:
        '''The main function
        '''
        last_electronics: ElectronicT = None

        # snapshot the initial frame
        self.snapshot()

        while self.continue_simulating():
            last_electronics, self.electronics = self.nuclear_movement(self.electronics)

            hamiltonian = self.propagate_electronics(last_electronics, self.electronics)

            self.surface_hopping(last_electronics, self.electronics, hamiltonian)

            # update
            self.time += self.dt
            self.timestep += 1

            if self.timestep % self.snapevery == 0:
                self.snapshot()

        # snapshot the last frame
        if self.timestep % self.snapevery != 0:
            self.snapshot()

        return self.tracer
