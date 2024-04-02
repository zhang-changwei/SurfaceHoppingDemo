import copy as cp
from typing import Optional, Dict, Any

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh

from plot import plot_qd_wavefunction, plot_adiabatic_potential_surface
from typings import DiabaticModelT

class QDSolver(object):
    '''
    The basic class for solving full quantum simulation.
    Can only solve 1D problem for now.
    '''
    def __init__(self,
                 ndvr: int = 1024,
                 xbound: float = 50.0,
                 plot: bool = True) -> None:
        # assert np.mod(ndvr, 2) == 0
        self.ndvr_ = ndvr
        self.xbound_ = xbound
        self.plot_ = plot
        self.ad_energy: ArrayLike
        self.ad_wfc: ArrayLike
        self.nonadiabatic_coupling: ArrayLike
        self._status: Dict
        self._result: Dict = {}

    def result(self) -> Dict:
        return self._result

    def compute(self,
                initial_wfc: ArrayLike,
                hamiltion: ArrayLike,
                nstate: int = 2,
                dt: float = 50,
                total_time: float = 20000,
                bounds: ArrayLike = [-5, 5],
                snapevery: int = 1) -> Dict[str, Any]:
        self._status = {'Enter': False, 'Exit': False, 'Init': True}

        coeff_t = np.zeros_like(initial_wfc)
        coeff_0 = np.zeros_like(initial_wfc)

        # propagation
        index = 1
        ndvr = self.ndvr_
        x = np.linspace(-self.xbound_, self.xbound_, num=ndvr, endpoint=False)
        dx = x[1] - x[0]

        eigen_energy, eigen_wfc = eigh(hamiltion)
        for i in range(ndvr):
            coeff_0[i::ndvr] = self.ad_wfc[i] @ initial_wfc[i::ndvr]
        coeff_0 = eigen_wfc.conj().T @ coeff_0

        coeff_ad = np.zeros([ndvr, nstate], dtype=np.complex128)
        charge_ad = np.zeros([2, nstate], dtype=np.float64)
        charge_ad_last = np.zeros([2, nstate], dtype=np.float64)

        for t in np.arange(0, total_time+1, dt):
            print(f'time {t}')
            propagator = np.diagflat(np.exp(-1j * eigen_energy * t))
            coeff_t = eigen_wfc @ propagator @ coeff_0

            # density
            # :math: c_{ad} = U^{dag} c_{DVR}
            for i in range(ndvr):
                coeff_ad[i] = self.ad_wfc[i].T @ coeff_t[i::ndvr]
            charge_ad[0] = np.sum(dx * np.abs(coeff_ad[:ndvr//2, :])**2, axis=0, dtype=np.float64)
            charge_ad[1] = np.sum(dx * np.abs(coeff_ad[-ndvr//2:,:])**2, axis=0, dtype=np.float64)

            self._check_status(np.abs(coeff_ad)**2 * dx, 
                               charge_ad, 
                               charge_ad_last, 
                               inbox=bounds)
            if self._status['Exit']:
                break
            charge_ad_last = cp.deepcopy(charge_ad)

            # print(charge_ad)
            # snapshot
            if self.plot_ and index % snapevery == 0:
                plot_qd_wavefunction(np.abs(coeff_ad), x, int(t))

            index += 1
        else:
            raise Exception('Charge density not converge. Please increase total time.')
        
        self._result['Time'] = t
        self._result['Charge'] = charge_ad
        plt.ioff()

        return self._result
    
    def oneshot(self,
                initial_wfc: ArrayLike,
                hamiltion: ArrayLike,
                nstate: int = 2,
                t: float = 10000) -> Dict[str, Any]:
        coeff_t = np.zeros_like(initial_wfc)
        coeff_0 = np.zeros_like(initial_wfc)

        # propagation
        ndvr = self.ndvr_
        x = np.linspace(-self.xbound_, self.xbound_, num=ndvr, endpoint=False)
        dx = x[1] - x[0]

        eigen_energy, eigen_wfc = eigh(hamiltion)
        for i in range(ndvr):
            coeff_0[i::ndvr] = self.ad_wfc[i] @ initial_wfc[i::ndvr]
        coeff_0 = eigen_wfc.conj().T @ coeff_0

        coeff_ad = np.zeros([ndvr, nstate], dtype=np.complex128)
        charge_ad = np.zeros([2, nstate], dtype=np.float64)

        propagator = np.diagflat(np.exp(-1j * eigen_energy * t))
        coeff_t = eigen_wfc @ propagator @ coeff_0

        # density
        # :math: c_{ad} = U^{dag} c_{DVR}
        for i in range(ndvr):
            coeff_ad[i] = self.ad_wfc[i].T @ coeff_t[i::ndvr]
        charge_ad[0] = np.sum(dx * np.abs(coeff_ad[:ndvr//2, :])**2, axis=0, dtype=np.float64)
        charge_ad[1] = np.sum(dx * np.abs(coeff_ad[-ndvr//2:,:])**2, axis=0, dtype=np.float64)

        # plot_qd_wavefunction(np.abs(coeff_ad), x, int(t))

        outbox_ngrid = 8
        outbox_thres = 1E-3
        charge = np.abs(coeff_ad)**2 * dx
        outbox_chg = np.sum(charge[:outbox_ngrid]) + np.sum(charge[-outbox_ngrid:])
        if outbox_chg > outbox_thres:
            raise Exception('Wavefunction out of boundary. Please increase xbound.')
        
        self._result['Time'] = t
        self._result['Charge'] = charge_ad

        return self._result

    def construct_hamiltion(self,
                            model: DiabaticModelT) -> ArrayLike:
        '''
        :return: hamiltion [ndvr x nstate, ndvr x nstate]
        '''
        ndvr: int = self.ndvr_
        nstate: int = model.nstate()
        mass: int = model.mass[0]
        hamiltion = np.zeros([ndvr * nstate, ndvr * nstate], dtype=np.float64)
        x = np.linspace(-self.xbound_, self.xbound_, num=ndvr, endpoint=False)
        dx: float = x[1] - x[0]

        # the kinetic energy operator
        # T = np.zeros([ndvr, ndvr], dtype=np.float64)
        # for i in range(ndvr):
        #     for j in range(i, ndvr):
        #         tmp = (-1)**(i-j) / (2 * mass * dx**2)
        #         if i == j:
        #             T[i,j] = tmp * np.pi**2 / 3
        #         else:
        #             T[i,j] = T[j,i] = tmp * 2 / (i-j)**2
        # for ist in range(nstate):
        #     hamiltion[ndvr*ist:ndvr*(ist+1), ndvr*ist:ndvr*(ist+1)] = T
        
        # the kinetic energy operator v2
        T = np.zeros([ndvr, ndvr], dtype=np.float64)
        L = 2 * self.xbound_
        N = ndvr + 1
        for i in range(1, N):
            for j in range(i, N):
                tmp = (-1)**(i-j) / (2 * mass * L**2) * (np.pi**2 / 2)
                if i == j:
                    T[i-1,j-1] = tmp * ((2 * ndvr**2 + 1)/3 - 
                                        1 / np.sin(i * np.pi / N)**2)
                else:
                    T[i-1,j-1] = T[j-1,i-1] = tmp * (1 / (np.sin(0.5 * np.pi * (i-j) / N))**2 -
                                                     1 / (np.sin(0.5 * np.pi * (i+j) / N))**2)
        for ist in range(nstate):
            hamiltion[ndvr*ist:ndvr*(ist+1), ndvr*ist:ndvr*(ist+1)] = T

        
        # the electronic part
        # And diag V to get adiabatic potential surface
        # the reference is only used to correct the phase
        self.ad_energy = np.zeros([ndvr, nstate], dtype=np.float64)
        self.ad_wfc = np.zeros([ndvr, nstate, nstate], dtype=np.float64)
        self.nonadiabatic_coupling = np.zeros([ndvr, nstate, nstate], dtype=np.float64)
        reference: Optional[ArrayLike] = None
        for i in range(ndvr):
            V = model.V([x[i]])
            dV = model.dV([x[i]])[0]
            hamiltion[i::ndvr,i::ndvr] += V
            en, wfc = eigh(V)

            if reference is not None:
                for ist in range(nstate):
                    if np.dot(reference[:,ist], wfc[:,ist]) < 0:
                        wfc[:,ist] *= -1.0
            self.ad_energy[i], self.ad_wfc[i] = en, wfc
            reference = wfc

            tmp = np.einsum('ip,ij,jq->pq', wfc, dV, wfc)
            for ist in range(nstate):
                for jst in range(ist):
                    dE = en[jst] - en[ist]
                    if abs(dE) < 1.0E-10:
                        dE = np.copysign(1.0E-10, dE)
                    self.nonadiabatic_coupling[i,ist,jst] = tmp[ist,jst] / dE
            self.nonadiabatic_coupling[i] -= self.nonadiabatic_coupling[i].T
        
        if self.plot_:
            plot_adiabatic_potential_surface(self.ad_energy, self.nonadiabatic_coupling, x)
        return hamiltion

    def construct_initial_wavefunction(self,
                                       model: DiabaticModelT,
                                       x0: float = -2.0,
                                       k0: float = 10.0,
                                       sigma: Optional[float] = None) -> ArrayLike:
        '''
        :return: wavefunction in DVR basis [ndvr x nstate]
        '''
        assert model.ndim_ == 1
        ndvr = self.ndvr_
        sigma = (sigma == None) and 10 / k0 or sigma
        x = np.linspace(-self.xbound_, self.xbound_, num=ndvr, endpoint=False)
        dx = x[1] - x[0]

        wfc = np.zeros([ndvr * model.nstate()], dtype=np.complex128)
        wfc[:ndvr] = np.exp(
            1j * k0 * x - (x - x0)**2/(4 * sigma**2)
        )
        # wfc[:ndvr] *= np.sqrt(1 / (sigma * np.sqrt(2*np.pi)))
        wfc[:ndvr] *= 1 / np.sqrt(dx * np.sum(np.abs(wfc[:ndvr])**2))

        # plot_qd_wavefunction(np.abs(wfc.reshape(2,ndvr).T), x, 0)
        # plt.ioff()

        return wfc
    
    def _check_status(self,
                      charge: ArrayLike,
                      charge_lr: ArrayLike,
                      charge_lr_last: ArrayLike,
                      outbox_ngrid: int = 8,
                      outbox_thres: float = 1E-3,
                      inbox: ArrayLike = [-5, 5],
                      inbox_thres: float = 0.01,
                      charge_change_thres: float = 1E-4,
                      charge_change_ratio_thres: float = 1E-3) -> None:
        '''
        :param: charge, ndvr x nstate array
        :param: charge_lr, 2 x nstate array
        :param: charge_lr_last, 2 x nstate array
        :param: outbox_ngrid
        '''
        if self._status['Init']:
            self._status['Init'] = False
            return

        outbox_chg = np.sum(charge[:outbox_ngrid]) + np.sum(charge[-outbox_ngrid:])
        if outbox_chg > outbox_thres:
            raise Exception('Wavefunction out of boundary. Please increase xbound.')
        
        idx_l = (inbox[0] + self.xbound_) / (self.xbound_ * 2) * self.ndvr_
        idx_r = (inbox[1] + self.xbound_) / (self.xbound_ * 2) * self.ndvr_
        idx_l, idx_r = max(0, round(idx_l)), min(self.ndvr_, round(idx_r))
        inbox_chg = np.sum(charge[idx_l:idx_r])
        if not self._status['Enter']:
            if inbox_chg > 0.5:
                self._status['Enter'] = True
            return
            
        dchg = np.abs(charge_lr - charge_lr_last)
        print(np.max(dchg), inbox_chg)
        if (# np.all(dchg[dchg>1E-10] / np.abs(charge_lr[dchg>1E-10]) < charge_change_ratio_thres) and
            np.all(dchg < charge_change_thres) and
            inbox_chg < inbox_thres):
            self._status['Exit'] = True
        return
    

