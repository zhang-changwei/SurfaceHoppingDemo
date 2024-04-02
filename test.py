import multiprocessing as mp
import numpy as np

from model import TullySimpleAvoidedCrossing as SAC
from model import TullyDualAvoidedCrossing as DAC
from model import TullyExtendedCouplingReflection as ECR
from qd import QDSolver
from sh import TrajectorySurfaceHopping, IniconGenNormal

def qd(x0, k0):
    kmax = np.sqrt((k0**2/4000+0.2) * 4000)
    sac = ECR()
    qdsolover = QDSolver(ndvr=4096, xbound=100, plot=False)
    hamilt = qdsolover.construct_hamiltion(sac)
    iniwav = qdsolover.construct_initial_wavefunction(sac, x0, k0)
    # result = qdsolover.oneshot(iniwav, hamilt,
    #                            nstate=sac.nstate(),
    #                            t=100000/k0) # vt = xbound/2
    result = qdsolover.compute(iniwav, hamilt,
                               nstate=sac.nstate(),
                               dt=5000/kmax,
                               total_time=1E5,
                               bounds=[-10, 5])

    print('k0 = {:f}, time = {:f}'.format(k0, result['Time']))
    print(result['Charge'])
    return result

    
def sh(x0, k0, ntraj=2000):
    sac = SAC()
    inicons = IniconGenNormal(x0, p0=k0)
    LR = 0
    UR = 0
    LT = 0
    UT = 0
    print('k0 = {:f}'.format(k0[0]))
    
    for itraj in range(ntraj):
        x0, k0 = inicons()
        # print(x0, k0)
        tshsolver = TrajectorySurfaceHopping(sac, x0, k0,
                                            initial_state=[1,0],
                                            dt=0.5,
                                            total_time=100000,
                                            active=0,
                                            MC=False,
                                            bounds=[[-8],[8]],
                                            snapevery=1)
        result = tshsolver.compute()
        pos = result[-1]['position']
        active = result[-1]['active']
        if (pos < 0 and active == 0): LR += 1
        if (pos < 0 and active == 1): UR += 1
        if (pos > 0 and active == 0): LT += 1
        if (pos > 0 and active == 1): UT += 1
        # print('{:d} {:d} {:d} {:d}'.format(LR, UR, LT, UT))
    LR /= ntraj
    UR /= ntraj
    LT /= ntraj
    UT /= ntraj
    print('{:f} {:f} {:f} {:f}'.format(LR, UR, LT, UT))
    return LR, UR, LT, UT


if __name__ == '__main__':
    x0 = -10.0
    fp = open('result/qdecr3.txt', 'w')
    fp.write('# k  LR  UR  LT  UT\n')
    for k0 in range(10,51):
        result = qd(x0, k0)
        fp.write('{:2d}  {:f} {:f} {:f} {:f}\n'.format(int(k0), 
                                                       result['Charge'][0,0], 
                                                       result['Charge'][0,1], 
                                                       result['Charge'][1,0], 
                                                       result['Charge'][1,1]))

    # results = []
    # pool = mp.Pool(processes=12)
    # x0 = np.array([-5.0])
    # fp = open('result/sh.txt', 'w')
    # fp.write('# k  LR  UR  LT  UT\n')
    # for k0 in range(4,51):
    #     result = pool.apply_async(sh, (x0, [k0]))
    #     results.append(result)
    # for k0, result in zip(range(4,51), results):
    #     LR, UR, LT, UT = result.get()
    #     fp.write('{:2d}  {:f} {:f} {:f} {:f}\n'.format(int(k0), LR, UR, LT, UT))
