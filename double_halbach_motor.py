"""
OpenMDAO component modeling a double Halbach array permanent-magnet ironless axial-flux motor.

Based on work by Kirsten P. Duffy, Thomas Tallerico, and Jeff Chin.
OpenMDAO Component by Kenneth T. Moore
"""
from __future__ import division, print_function
from six.moves import range
from math import pi

import numpy as np

from openmdao.api import ExplicitComponent, AnalysisError

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
        self.npole = 24            # Number of magnetic pole pairs
        self.nm = 8                # Number of magnets in a pole pair
        self.Br = 1.35             # Permanent magnet remanence flux (1T = 1N*m/Amp)
        self.rho_mag = 7500.       # Magnet material mass density (kg/m**3)

        # Stator Material and Physical Constant Parameters
        self.ag = .001              # Air gap between rotor and stator. (m)
        self.nphase = 3             # Number of Phases
        self.rho_stator = 8940.     # Density of stator material (Cu) (kg/m**3)
        self.Imax = 6.0             # Max current for 1 wire (Amp)
        self.R = 1.68e-8            # Wire resistance (ohms * m)
        self.minwall = .1 * .0254   # Min thickness of wire cores (tooth) at inner r (m)
        self.cfill = .5             # Copper fill percentage

        self.rwire = rwire = .000127 * 0.5
        self.Awire = pi*(rwire)**2  # Square volume taken up by 1 wire (m**2).

        # Discretization
        self.nr = 10
        self.ntime = 1              # Currently only used as a sanity check.

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

        self.add_input('magnet_width', val=.005, units='m',
                       desc='Magnet width (tangential).')

        self.add_input('magnet_depth', val=.005, units='m',
                       desc='Magnet depth (axial).')

        self.add_input('yg', val=.0035 , units='m',
                       desc='Distance between rotor surface and stator centerline.')

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
        rwire = self.rwire

        ntime = self.ntime
        nr = self.nr

        # Thickness of coil
        yw = 2.0 * (yg - ag)

        # Area available for wire (I-beam area)
        A = (2.0*R0*pi/(nphase*npole) - minwall) * yw * 0.5

        # Width of coil
        xw = A / yw

        # Number of wires per phase
        nw = np.floor(A * cfill/Awire)

        # Estimate Minimum Length of all Wire
        wire_length = nw*(2.0*(RF-R0) + (2.0*RF*pi + 2*R0*pi)/(npole*nphase))

        # Discretize in radial direction
        dr = (RF - R0) / nr
        r = R0 + 0.5*dr + dr*np.arange(nr)

        # Discretize in y: just need start and end
        y = np.array([-yg + ag, yg - ag])

        # Discretize in time
        omega = RPM / 30. * pi
        if ntime > 1:
            t_end = 2.0 * pi / (npole * omega)
            dt = t_end / (ntime-1)
            t = dt*np.arange(ntime)
        else:
            t = np.array([0.0])

        # Magnet mass
        M_magnet = self.rho_mag * xm * ym * (RF - R0) * npole * nm * 2.0

        PR = np.empty((ntime, ))
        F = np.empty((2*nphase, ))
        T_coil = np.empty((nr, ))
        T_total = np.empty((ntime, ))

        # Integrate over time.
        for z in range(ntime):
            t_z = t[z]

            # Define Coil Currents in each coil direction
            I = Imax * np.cos(npole * omega * t_z - (2.0 * pi / nphase) * np.arange(nphase))
            I = np.append(I, -I)     # current moves in and out of plane
            J = I / (Awire*nw)       # Amps/m**2 Current Density

            # save peak current density in Amps/mm**2
            self.current_density = Imax / (Awire*nw) * .000001

            # Calculate resistance.
            PR[z] = npole * R * wire_length * np.sum(I**2)

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
                if e > 1.0:
                    msg = "Magnets are too large for the allotted space."
                    raise AnalysisError(msg)

                # Define Coil Currents and x start points.
                # Equally spaced coils centerpoints + time adjust.
                delta = xp / (2.0*nphase)
                xws = delta * np.arange(2*nphase) + 0.5*delta + x_adjust

                # Reorder to A+ C- B+ A- C+ B-
                xws = xws[np.array([0, 2, 4, 3, 5, 1])]

                print(xp, xws)

                # Intermediate terms for Flux calculation.
                k = 2.0 * pi / xp
                Bterm = 2.0 * Br * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e*pi/nm) * nm/pi
                sinh_ky = np.sinh(k*y)

                # Force is calculated one coil at a time.
                for n in range(2*nphase):   # Current always symmetric about center
                    xws_n = xws[n]

                    # x start and end values for current coil.
                    x = np.array([xws_n, xws_n + xw])

                    # Analytically integrate the flux over x and y
                    By = -Bterm * (np.sin(k*x[-1]) - np.sin(k*x[0])) * (sinh_ky[-1] - sinh_ky[0]) / k**2

                    # Flux to Force
                    F[n] = J[n] * By * dr

                # Estimate the flux squared for Eddy Loss calculation.
                # Max flux occurs at
                # - innermost radius
                # - at x (and phase) where cos(kx) is 1
                # - at y on either edge
                # sinh(x)**2 + cosh(x)**2 = cosh(2x)
                if q == 0:
                    B_sq_max_est = Bterm**2 * np.cosh(2.0 * k * y[0])

                # Torque for each coil at radius r.
                T_coil[q] = np.abs(r_q * np.sum(F))

            # Torque from each radii.
            T_total[z] = np.sum(T_coil) * npole
            print(T_coil)

        # Power at Rpm. (W)
        P = np.sum(T_total) * omega / ntime

        # Estimated resistive losses. (W)
        PR = np.sum(PR) / ntime

        # Eddy loss calculcation. (W)
        vol_cond = 2.0 * (RF - R0) * Awire * nw * nphase * npole
        Pe = 0.5 * (np.pi * npole * RPM/60.0 * 2.0 * rwire)**2 * vol_cond / R * B_sq_max_est

        efficiency = (P - PR - Pe) / P

        # Mass sum. (kg)
        M = wire_length * Awire * self.rho_stator * npole * nphase + M_magnet

        # Power Density (converted to kW/kg)
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

    #for jj in np.logspace(0.3, 4, 30):
        #prob.model.comp.nx = int(jj)
        #prob.run_model()