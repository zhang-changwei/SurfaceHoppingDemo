import copy as cp
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh

class ElectronicModel_(object):
    def __init__(self, 
                 representation: str = 'adiabatic',
                 reference: Any = None,
                 nstate: int = 2,
                 ndim: int = 1) -> None:
        self.ndim_ = ndim
        self.nstate_ = nstate

        self._reprenstation = representation
        self._reference = reference
        self._position: ArrayLike # ndim

        self._hamiltonian: ArrayLike # [nstate, nstate]
        self._force: ArrayLike # [nstate, ndim]
        self._derivative_coupling: ArrayLike # [nstate, nstate, ndim]

    def ndim(self) -> int:
        return self.ndim_
    
    def nstate(self) -> int:
        return self.nstate_
    
    def representation(self) -> str:
        return self._reprenstation
    
    def reference(self) -> Any:
        return self._reference
    
    def position(self) -> ArrayLike:
        return self._position
    
    def hamiltonian(self) -> ArrayLike:
        return self._hamiltonian
    
    def force(self, state: int = 0) -> ArrayLike:
        '''Return the force on a given state'''
        return self._force[state,:]
    
    def derivative_coupling(self, statei: int = 0, statej: int = 1) -> ArrayLike:
        '''Return the derivative coupling between two states'''
        return self._derivative_coupling[statei,statej,:]

    def NAC_matrix(self, velocity: ArrayLike) -> ArrayLike:
        '''Return the nonadiabatic coupling matrix for a given velocity vector.
        '''
        return np.einsum('ijx,x->ij', self._derivative_coupling, velocity)
    
    def compute(self, x: ArrayLike, reference: Any = None) -> None:
        raise NotImplementedError
    
    def update(self, x: ArrayLike) -> 'ElectronicModel_':
        '''
        :return: a shallow copy of electronicmodel at positon x
        '''
        out = cp.copy(self)
        out.compute(x, self.reference())
        return out

    def clone(self) -> 'ElectronicModel_':
        return cp.deepcopy(self)
    
    def as_dict(self):
        pass


class DiabaticModel_(ElectronicModel_):
    '''
    Base class to handle model problems given in
    simple diabatic forms.

    To derive from DiabaticModel_, the following functions
    must be implemented:
        - def V(self, X: ArrayLike) -> ArrayLike
          V(x) should return an ndarray of shape (nstates, nstates)
        - def dV(self, X: ArrayLike) -> ArrayLike
          dV(x) shoudl return an ndarry of shape (nstates, nstates, ndim)
    '''

    def __init__(self, 
                 representation: str = 'adiabatic', 
                 reference: Any = None, 
                 nstate: int = 2, 
                 ndim: int = 1) -> None:
        super().__init__(representation, reference, nstate, ndim)

    def compute(self, x: ArrayLike, reference: Any = None) -> None:
        self._position = x
        V = self.V(x)
        dV = self.dV(x)

        coeff, energies = self._compute_basis_states(V, reference)
        self._derivative_coupling = self._compute_derivative_coupling(dV, coeff, energies)
        self._force = self._compute_force(dV, coeff)
        
        # update reference
        self._reference = coeff
        self._hamiltonian = energies

    def _compute_basis_states(self, V: ArrayLike, reference: Any = None):
        '''Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        :return: coeff or unitary matix, n x n array
        :return: eigenenergy or V, n x n array
        '''
        if self.representation() == 'adiabatic':
            energies, coeff = eigh(V)
            if reference is not None:
                for ist in range(self.nstate()):
                    if np.dot(reference[:,ist], coeff[:,ist]) < 0:
                        coeff[:,ist] *= -1.0
            return coeff, np.diag(energies)
        elif self._representation == 'diabatic':
            return np.eye(self.nstates(), dtype=np.float64), V
        else:
            raise Exception('Unrecognized run mode')

    def _compute_force(self, dV: ArrayLike, coeff: ArrayLike) -> ArrayLike:
        ''':return: force, n x ndim array'''
        out = -np.einsum('ip,xij,jp->px', coeff, dV, coeff)
        return out
    
    def _compute_derivative_coupling(self, dV: ArrayLike, coeff: ArrayLike, energies: ArrayLike) -> ArrayLike:
        ''':return: nonadiabatic coupling vector, n x n x ndim array'''
        nst = self.nstate()
        out = np.zeros([nst, nst, self.ndim()], dtype=np.float64)
        if self.representation() == 'diabatic':
            return out
        
        out = np.einsum('ip,xij,jq->pqx', coeff, dV, coeff)
        for i in range(nst):
            for j in range(i):
                dE = energies[j,j] - energies[i,i]
                if abs(dE) < 1.0E-10:
                    dE = np.copysign(1.0E-10, dE)
                out[i,j,:] /= dE
                out[j,i,:] /= -dE
            out[i,i,:] = 0.0
        return out
    
    def V(self, x: ArrayLike) -> ArrayLike:
        return NotImplementedError
    
    def dV(self, x: ArrayLike) -> ArrayLike:
        return NotImplementedError