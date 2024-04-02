from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def plot_qd_wavefunction(wfc: ArrayLike, x: ArrayLike, time: int) -> None:
    nstate = wfc.shape[1]

    plt.ion()
    fig: Figure = plt.gcf()
    plt.clf()
    for ist in range(nstate):
        ax = fig.add_subplot(nstate, 1, ist+1)
        ax.plot(x, wfc[:,ist], '-')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1)
    plt.title('{:d} fs'.format(time))
    plt.pause(0.01)
    plt.show()

def plot_adiabatic_potential_surface(energy: ArrayLike,
                                     nonadiabatic_coupling: ArrayLike,
                                     x: ArrayLike) -> None:
    nstate = energy.shape[1]

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, energy[:,0], '-', label='$E_0$')
    ax.plot(x, energy[:,1], '-', label='$E_1$')
    ax.plot(x, nonadiabatic_coupling[:,0,1], '-', label=r'$d_{01}$')
    ax.set_xlim(-10, 10)

    plt.legend()
    plt.show()
    fig.savefig('adiabatic_energy_surface.png')