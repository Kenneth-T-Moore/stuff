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
        self.Br = 1.35             # Permanent magnet remanence flux (1T = 1N*m/Amp)
        self.rho_mag = 7500.       # Magnet material mass density (kg/m**3)

        # Stator Material and Physical Constant Parameters
        self.rho_stator = 8940.     # Density of stator material (Cu) (kg/m**3)
        self.minwall = .1 * .0254   # Min thickness of wire cores (tooth) at inner r (m)
        self.cfill = .5             # Copper fill percentage

        self.rwire = rwire = .000255 * 0.5

        # Discretization
        self.nr = 10
        self.ntime = 1              # Currently only used as a sanity check.

    def initialize(self):
        """
        Declare options accessable via the self.options dictionary.
        """
        self.options.declare('overlap', False,
                             desc='Set to True for overlapping windings. Default is concentrated.')

        self.options.declare('num_poles', 24,
                             desc='Number of magnetic pole pairs.')

        self.options.declare('num_magnets', 8,
                             desc='Number of magnets in a pole pair.')

        self.options.declare('num_phases', 3,
                             desc='Number of phases in motor.')

        # Define Coil Currents and x start points.
        # Defaut values come from the motor in the Duffy paper, 90 degree load.
        # Note, these are normalized values.
        nphase = 3
        delta = 1.0 / (2.0*nphase)
        xws_norm = delta * np.arange(2*nphase)
        # Reorder to A+ C- B+ A- C+ B-
        xws_norm = xws_norm[np.array([0, 2, 4, 3, 5, 1])]

        self.options.declare('coil_locations', xws_norm,
                             desc='Coil centerpoint locations. Should be a vector of length 2*num_phases, '
                             'normalized on the x axis of the coil.')

    def setup(self):
        """
        Declare input and output for this component, and declare derivatives.
        """

        # Inputs
        # ------

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

        self.add_input('coil_thickness', val=.0066 , units='m',
                       desc='Coil Thickness.')

        self.add_input('Imax', val=6.0 , units='A',
                       desc='Motor peak current.')

        self.add_input('resistivity', val=1.68e-8 , units='ohm*m',
                       desc='Coil resistivity, may come from a temperature model in another component.')

        self.add_input('air_gap', .001, units='m',
                       desc = 'Air gap between rotor and stator.')

        # Outputs
        # -------

        self.add_output('power_ideal', units='kW',
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
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        """
        Integrate magnetic flux between Halbach array over axial, circumferential, radial, and
        temporal dimension.
        """
        R0 = inputs['inner_radius']
        RF = inputs['outer_radius']
        xm = inputs['magnet_width']
        ym = inputs['magnet_depth']
        yw = inputs['coil_thickness']
        ag = inputs['air_gap']
        Imax = inputs['Imax']
        RPM = inputs['RPM']
        resistivity = inputs['resistivity']

        nm = self.options['num_magnets']
        npole = self.options['num_poles']
        nphase = self.options['num_phases']
        xws_norm = self.options['coil_locations']

        minwall = self.minwall
        cfill = self.cfill
        Br = self.Br
        rwire = self.rwire

        ntime = self.ntime
        nr = self.nr

        # Area of a single wire strand.
        Awire = pi*(rwire)**2  # Square volume taken up by 1 wire (m**2).

        # Back calculate from thickness of coil
        yg = 0.5 * yw + ag

        # Area available for wire (I-beam area)
        A = (2.0*R0*pi/(nphase*npole) - minwall) * yw * 0.5

        # Width of coil
        xw = A / yw

        # Number of wires per phase
        # (No longer used in calculation, but kept for reference.)
        nw = np.floor(A * cfill/Awire)

        # Area of all wires in a single coil.
        Awires = A * cfill

        # Estimate Minimum Length of all Wire
        # (Assumes 1 wire per Phase coils joined on inner radius)
        if self.options['overlap']:
            wire_length = 2.0*(RF-R0)*npole + pi*(RF + R0)
        else:
            wire_length = 2.0*(RF-R0)*npole + 2.0*pi*(RF + R0)/nphase

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
        M_magnet = self.rho_mag * xm * ym * (RF - R0) * npole * nm * 2

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
            J = I / Awires      # Amps/m**2 Current Density

            # save peak current density in Amps/mm**2
            self.current_density = Imax / Awires * .000001

            # Calculate resistance.
            phase_R = wire_length * resistivity / (A * cfill)
            PR[z] = nphase * 0.5 * phase_R * np.sum(I**2)

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
                if e > 1.0 + 1e-8:
                    msg = "Magnets are too large for the allotted space."
                    raise AnalysisError(msg)

                # Define Coil Currents and x start points.
                xws = xws_norm * xp + x_adjust

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

        # Power at Rpm. (W)
        P = np.sum(T_total) * omega / ntime

        # Estimated resistive losses. (W)
        PR = np.sum(PR) / ntime

        # Eddy loss calculcation. (W)
        vol_cond = 2.0 * (RF - R0) * Awires * nphase * npole
        Pe = 0.5 * (np.pi * npole * RPM/60.0 * 2.0 * rwire)**2 * vol_cond / resistivity * B_sq_max_est

        efficiency = (P - PR - Pe) / P

        # Mass sum. (kg)
        M = wire_length * Awires * self.rho_stator * nphase + M_magnet

        # Power Density (converted to kW/kg)
        power_density = P/M * 0.001

        # Convert all outputs to kg, where applicable
        outputs['power_ideal'] = P * 0.001
        outputs['power_density'] = power_density
        outputs['resistive_loss'] = PR * 0.001
        outputs['eddy_current_loss'] = Pe * 0.001
        outputs['efficiency'] = efficiency


class DoubleHalbachMotorThermalComp(ExplicitComponent):
    """
    Component that models the thermal properties of the Halbach Array motor, computing the stator
    temperature and the coil resistivity assuming a copper conductor.

    This is tailored for use in an aircraft, so air properties come from freestream flight
    conditions.
    """

    def setup(self):
        """
        Declare input and output for this component, and declare derivatives.
        """

        # Inputs
        self.add_input('inner_radius', val=6.125 * 0.0254, units='m',
                       desc='Rotor inner radius.')

        self.add_input('coil_thickness', val=.0066 , units='m',
                       desc='Coil Thickness.')

        self.add_input('resistive_loss', val=0.0, units='kW',
                        desc='Resistive loss.')

        self.add_input('eddy_current_loss', val=0.0, units='kW',
                        desc='Eddy current loss.')

        self.add_input('rho_air', val=1.0, units='kg/m**3',
                        desc='Density for air.')

        self.add_input('mu_air', val=1.7e-5, units='N*s/m**2',
                        desc='Dynamic viscosity for air.')

        self.add_input('Cp_air', val=1005, units='J/(kg*degK)',
                        desc='Specific heat capactity at constant pressure for air.')

        self.add_input('k_air', val=2.4e-2, units='W/(m*degK)',
                       desc='Thermal conductivity of air.')

        self.add_input('T_air', val=272.0, units='degK',
                       desc='Temperature for air.')

        self.add_input('V_air', val=70.0, units='m/s',
                        desc='Velocity of aircraft.')

        # Outputs
        self.add_output('stator_temperature', units='degK',
                        desc='Temperature of the stator. 420 degK is rating of winding insulation.')

        self.add_output('resistivity', units='ohm*m',
                        desc='Coil resistivity at computed temperature.')

        # Derivatives
        # -----------
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        """
        Compute thermal properties.
        """
        r1 = inputs['inner_radius']
        loss_stator = inputs['resistive_loss'] + inputs['eddy_current_loss']
        rho = inputs['rho_air']
        mu = inputs['mu_air']
        Cp = inputs['Cp_air']
        T = inputs['T_air']
        V = inputs['V_air']
        k = inputs['k_air']
        coil_tk = inputs['coil_thickness']

        Re = rho * V / mu
        Pr = mu * Cp / k
        Hconv = 0.644 * k * Pr**(1.0/3.0) * np.sqrt(Re)
        T_stator = loss_stator / (np.sqrt(coil_tk) * 2.0 * pi * Hconv * r1) + T

        outputs['stator_temperature'] = T_stator

        # Copper Resistivity (Ohm m)
        outputs['resistivity'] = 1.75e-8 * (1.0 + 3.81e-3*(T_stator - 293))

    def compute_partials(self, inputs, partials):
        r1 = inputs['inner_radius']
        loss_stator = inputs['resistive_loss'] + inputs['eddy_current_loss']
        rho = inputs['rho_air']
        mu = inputs['mu_air']
        Cp = inputs['Cp_air']
        T = inputs['T_air']
        V = inputs['V_air']
        k = inputs['k_air']
        coil_tk = inputs['coil_thickness']

        Pr = mu * Cp / k
        Re = rho * V / mu
        Hconv = 0.644 * k * Pr**(1.0/3.0) * np.sqrt(Re)

        dHconv_drho = 0.644 * k * Pr**(1.0/3.0) * 0.5 / np.sqrt(Re) * V / mu
        dHconv_dmu = 0.644 * k * (Pr**(-2.0/3.0) / 3.0 * Cp / k * np.sqrt(Re) +
                                  Pr**(1.0/3.0) * 0.5 / np.sqrt(Re) * (-rho * V / mu**2))
        dHconv_dCp = 0.644 * k * Pr**(-2.0/3.0) / 3.0 * mu / k * np.sqrt(Re)
        dHconv_dV = 0.644 * k * Pr**(1.0/3.0) * 0.5 / np.sqrt(Re) * rho / mu
        dHconv_dk = 0.644 * np.sqrt(Re) * (Pr**(1.0/3.0) + k * Pr**(-2.0/3.0) / 3.0 * -mu * Cp / k**2)

        denom = np.sqrt(coil_tk) * 2.0 * pi * Hconv * r1

        partials['stator_temperature', 'inner_radius'] = -loss_stator * np.sqrt(coil_tk) * 2.0 * pi * Hconv / denom**2
        partials['stator_temperature', 'coil_thickness'] = -0.5 * loss_stator / (coil_tk**1.5 * 2.0 * pi * Hconv * r1)

        term = -loss_stator / (np.sqrt(coil_tk) * 2.0 * pi * Hconv**2 * r1)
        partials['stator_temperature', 'rho_air'] = dHconv_drho * term
        partials['stator_temperature', 'mu_air'] = dHconv_dmu * term
        partials['stator_temperature', 'Cp_air'] = dHconv_dCp * term
        partials['stator_temperature', 'V_air'] = dHconv_dV * term
        partials['stator_temperature', 'k_air'] = dHconv_dk * term
        partials['stator_temperature', 'T_air'] = 1.0

        partials['stator_temperature', 'resistive_loss'] = 1.0 / denom
        partials['stator_temperature', 'eddy_current_loss'] = 1.0 / denom

        wrts = ['inner_radius', 'resistive_loss', 'eddy_current_loss', 'rho_air', 'mu_air', 'Cp_air',
                'T_air', 'V_air', 'k_air', 'coil_thickness']
        for wrt in wrts:
            partials['resistivity', wrt] = 1.75e-8 * 3.81e-3 * partials['stator_temperature', wrt]


if __name__ == "__main__":

    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    prob.model.add_subsystem('comp', DoubleHalbachMotorComp())

    prob.setup()
    prob.run_model()

    for name in ['power_ideal', 'power_density', 'resistive_loss', 'eddy_current_loss', 'efficiency']:
        print(name, prob['comp.' + name])

    #for jj in np.logspace(0.3, 4, 30):
        #prob.model.comp.nx = int(jj)
        #prob.run_model()