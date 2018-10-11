"""
OpenMDAO component modeling a double Halbach array permanent-magnet ironless axial-flux motor.

Based on work by Kirsten P. Duffy.
OpenMDAO Component by Kenneth T. Moore
"""
from __future__ import division, print_function
from six.moves import range
from math import pi

import numpy as np

from openmdao.api import ExplicitComponent

CITATION = """@PROCEEDINGS{
              Duffy2016,
              AUTHOR = "K. Duffy",
              TITLE = "Optimizing Power Density and Efficiency of a Double-Halbach Array "
                      "Permanent-Magnet Ironless Axial-Flux Motor",
              BOOKTITLE = "52nd AIAA/SAE/ASEE Joint Propulsion Conference",
              PUBLISHER = "AIAA",
              YEAR = "2016",
              MONTH = "JULY",
              }"""


class DoubleHalbachMotorComp(ExplicitComponent):
    """
    Component that models a double Halbach array permanent-magnet ironless axial-flux motor.
    """

    def __init__(self, **kwargs):
        """
        Initialize attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(DoubleHalbachMotorComp, self).__init__()
        self._kwargs = kwargs
        self.cite = CITATION

        # Rotor Material and Physical Constant Parameters
        self.lm1 = 10.0            # Magnet length (radial)
        self.npole = 24            # Number of magnetic pole pairs
        self.nm = 4                # Number of magnets in a pole pair
        self.Br = 1.0              # Permanent magnet remanence flux (1T = 1N*m/Amp)
        self.rho_mag = 7300.       # Magnet material mass density (kg/m**3)

        # Stator Material and Physical Constant Parameters
        self.ag = .001              # Air gap between rotor and stator. (m)
        self.nphase = 3             # Number of Phases
        self.rho_stator = 8960.     # Density of stator material (Cu) (kg/m**3)
        self.Imax = 6.0             # Max current for 1 wire (Amp)
        self.R = 34.1               # Wire resistance (ohms/Km)
        self.minwall = .1 * .0254   # Min thickness of wire cores (tooth) at inner r (m)
        self.cfill = .5             # Copper fill percentage

        rwire = .0004
        self.Awire = pi*(rwire)**2  # Square volume taken up by 1 wire (m**2)

        # Discretization
        self.nr = 10
        self.nx = 10
        self.ny = 10
        self.ntime = 100

    def setup(self):
        """
        Declare input and output for this component, and declare derivatives.
        """

        # Design Variables
        # ----------------

        self.add_input('inner_radius', val=6.125 * 0.0254, units='m',
                       desc='Rotor inner radius.')

        self.add_input('outer_radius', val=6.875 * 0.0254, units='m',
                       desc='Rotor outer radius.')

        self.add_input('RPM', val=8600., units='rpm',
                       desc='Motor rotational speed.')

        self.add_input('magnet_width', val=3./8. * 0.0254, units='m',
                       desc='Magnet width (tangential).')

        self.add_input('magnet_depth', val=3./16. * 0.0254, units='m',
                       desc='Magnet depth (axial).')

        self.add_input('yg', val=.0035 , units='m',
                       desc='Distance between rotor and stator cenerline.')

        # Outputs
        # -------

        self.add_output('power', units='kW',
                        desc='Motor output power.')

        self.add_output('power_density', units='kW/kg',
                        desc='Motor power density.')

        self.add_output('resistive_loss', units='kW',
                        desc='Resistive loss.')

        self.add_output('eddy_current_loss', units='kW',
                        desc='Eddy current loss.')

        self.add_output('efficiency',
                        desc='Total efficiency of the motor (accounting for all losses).')

        # Derivatives
        # -----------

    def compute(self, inputs, outputs):
        """
        Integrate magnetic flux between Halbach array over axial, circumferential, radial, and
        temporal dimension.
        """
        R0 = inputs['inner_radius']
        RF = inputs['outer_radius']
        xm = inputs['magnet_width']
        ym = inputs['magnet_depth']
        yg = inputs['yg']
        RPM = inputs['RPM']

        minwall = self.minwall
        ag = self.ag
        nm = self.nm
        npole = self.npole
        nphase = self.nphase
        cfill = self.cfill
        Awire = self.Awire
        Imax = self.Imax
        R = self.R
        Br = self.Br

        ntime = self.ntime
        nx = self.nx
        ny = self.ny
        nr = self.nr

        # Thickness of coil
        yw = 2.0 * (yg - ag)

        # Area available for wire (I-beam area)
        A = (2.0*R0*pi/(nphase*npole) - minwall) * yw * 0.5

        # Width of coil
        xw = A / yw

        # Number of wires per phase
        nw = np.floor(A * cfill/Awire)

        # Estimate Minimum Length of 1 Wire
        wire_length = nw*(2.0*(RF-R0) + (2.0*RF*pi + 2*R0*pi)/(npole*nphase))

        # Discretize in radial direction
        dr = (RF - R0) / nr
        r = R0 + 0.5*dr + dr*np.arange(nr)

        # Discretize in y
        dy = 2.0 * (yg - ag) / (ny-1)
        y = -yg + ag + dy*np.arange(ny)

        # Discretization of x depends on r, so it is done later.
        dx = xw / (nx-1)

        # Discretize in time
        omega = RPM / 30. * pi
        t_end = 2.0 * pi / (npole * omega)
        dt = t_end / (ntime-1)
        t = dt*np.arange(ntime)

        # Magnet mass
        M_magnet = self.rho_mag * xm * ym * (RF - R0) * npole * nm * 2.0

        PR = np.empty((ntime, ))
        F = np.empty((2*nphase, ))
        T_coil = np.empty((nr, ))
        T_total = np.empty((ntime, ))
        Bmax = 0.0

        # Integrate over time.
        for z in range(ntime):
            t_z = t[z]

            # Define Coil Currents in each coil direction
            I = Imax * np.cos(npole * omega * t_z + (2 * pi / nphase ) * np.arange(nphase))
            I = np.append(I, -I)     # current moves in and out of plane
            J = I / Awire            # Amps/m^2 Current Density

            # Calculate resistance.
            PR[z] = npole * R * wire_length * .001 * np.sum(I**2)

            # Integrate over radius (dr)
            for q in range(nr):
                r_q = r[q]

                # Adjust position of rotor relative to stator at given point in time.
                x_adjust = omega * r_q * t_z

                # Define Halbach Array parameters.

                # m length of one pole pair at radius.
                xp = 2.0 * pi * r_q / npole

                # Percent of material that is magnet in x direction.
                e = (nm * xm) / xp

                # Define Coil Currents and x start points.
                xws = 0.5 * (xp + xm) - xw + xp / nphase * np.arange(nphase) + x_adjust
                xws = np.append(xws, xws - xw + xp / nphase)   # Negative versions of each coil

                # Intermediate terms for Flux calculation.
                k = 2.0 * pi / xp
                Bterm = 2.0 * Br * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e*pi/nm) * nm/pi
                cosh_ky = np.cosh(k*y)
                sinh_ky = np.sinh(k*y)

                # Force is calculated one coil at a time.
                for n in range(2*nphase):   # Current always semetric about center
                    xws_n = xws[n]

                    # x values for current coil.
                    x = xws_n + dx * np.arange(nx)

                    # Integration over x and y can be done in one vector operation.
                    cos_kx = np.cos(k*x)
                    By0 = Bterm * np.outer(cos_kx, cosh_ky)
                    Bx0 = Bterm * np.outer(cos_kx, sinh_ky)

                    B = np.max((Bx0**2 + By0**2)**0.5)
                    Bmax = max(B, Bmax)

                    F[n] = J[n] * np.sum(By0) * dx * dy * dr

                # Sum Torque from all coils at radius r.
                T_coil[q] = np.abs(r_q * cfill * np.sum(F))

            # Sum torque from all radii.
            T_total[z] = np.sum(T_coil) * 2.0 * npole

        # Power at Rpm. (W)
        P = np.sum(T_total) * omega / ntime

        # Estimated resistive losses. (W)
        PR = np.sum(PR) / ntime

        # Eddy loss calculcation. (W)
        Pe = pi*(RPM/60.)**2 * npole * (1000./R) * Awire * nw * (RF-R0) * nphase * npole * 2.0 * Bmax

        efficiency = P/(P+PR+Pe)

        # Mass sum. (kg)
        M = wire_length * Awire * self.rho_stator * npole * nphase + M_magnet

        power_density = P/M * 0.001

        outputs['power'] = P * 0.001
        outputs['power_density'] = power_density
        outputs['resistive_loss'] = PR * 0.001
        outputs['eddy_current_loss'] = Pe * 0.001
        outputs['efficiency'] = efficiency


if __name__ == "__main__":

    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    prob.model.add_subsystem('comp', DoubleHalbachMotorComp())

    prob.setup()
    prob.run_model()

    for name in ['power', 'power_density', 'resistive_loss', 'eddy_current_loss', 'efficiency']:
        print(name, prob['comp.' + name])
