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

CITATION1 = """@PROCEEDINGS{
               Duffy2016,
               AUTHOR = "K. Duffy",
               TITLE = "Optimizing Power Density and Efficiency of a Double-Halbach Array "
                       "Permanent-Magnet Ironless Axial-Flux Motor",
               BOOKTITLE = "52nd AIAA/SAE/ASEE Joint Propulsion Conference",
               PUBLISHER = "AIAA",
               YEAR = "2016",
               MONTH = "JULY",
               }"""

CITATION2 = """@PROCEEDINGS{
               Tallerico2018,
               AUTHOR = "T. Tallerico, J. Chin, Z. Cameron",
               TITLE = "Optimization of an Air Core Dual Halbach Array Axial Flux Rim Drive for "
                       "Electric Aircraft",
               BOOKTITLE = "AIAA Aviation Forum",
               PUBLISHER = "AIAA",
               YEAR = "2018",
               MONTH = "JUNE",
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
        super(DoubleHalbachMotorComp, self).__init__(**kwargs)
        self.cite = CITATION1

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

        self.add_input('Ipeak', val=6.0 , units='A',
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

        self.add_output('rotor_mass', units='kg',
                        desc='Mass of the rotor.')

        self.add_output('stator_mass', units='kg',
                        desc='Mass of the stator.')

        # Derivatives
        # -----------
        self.declare_partials(of='rotor_mass', wrt=['inner_radius', 'outer_radius', 'magnet_depth', 'magnet_width'])
        self.declare_partials(of='stator_mass', wrt=['inner_radius', 'outer_radius', 'coil_thickness'])
        self.declare_partials(of='resistive_loss', wrt=['Ipeak', 'coil_thickness', 'inner_radius', 'outer_radius', 'resistivity'])
        self.declare_partials(of=['power_ideal', 'power_density', 'eddy_current_loss', 'efficiency'],
                              wrt='*')

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
        Ipeak = inputs['Ipeak']
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

        if RF <= R0:
            msg = "Inner radius should be less than outer radius."
            raise AnalysisError(msg)

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
        # nw = np.floor(A * cfill/Awire)

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

        # save peak current density in Amps/mm**2
        self.current_density = Ipeak / Awires * .000001

        # Calculate resistance.
        phase_R = wire_length * resistivity / (A * cfill)
        PR = nphase * 0.5 * phase_R * (Ipeak)**2

        T_coil = np.empty((nr, ), dtype=inputs._data.dtype)
        T_total = np.empty((ntime, ), dtype=inputs._data.dtype)

        # Integrate over time.
        for z in range(ntime):
            t_z = t[z]

            # Define Coil Currents in each coil direction
            I = Ipeak * np.cos(npole * omega * t_z - (2.0 * pi / nphase) * np.arange(nphase))
            I = np.append(I, -I)     # current moves in and out of plane
            J = I / Awires           # Amps/m**2 Current Density

            # Integrating over radius (dr) at all points simultaneously via vector operation.

            # Adjust position of rotor relative to stator at given point in time.
            x_adjust = omega * r * t_z

            # m length of one pole pair at radius.
            xp = 2.0 * pi * r / npole

            # Percent of material that is magnet in x direction.
            e = (nm * xm) / xp

            if e[0] > 1.0 + 1e-8:
                msg = "Magnets are too large for the allotted space."
                raise AnalysisError(msg)

            for q in range(nr):
                r_q = r[q]

                # Define Coil Currents and x start points.
                xws = xws_norm * xp[q] + x_adjust[q]

                # Intermediate terms for Flux calculation.
                k = 2.0 * pi / xp[q]
                Bterm = 2.0 * Br * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e[q]*pi/nm) * nm/pi
                sinh_ky = np.sinh(k*y)

                # Force is calculated for all coils, current always symmetric about center
                # Analytically integrate the flux over x and y
                By = Bterm * (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (sinh_ky[-1] - sinh_ky[0]) / k**2

                # Flux to Force
                F = J * By * dr

                # Estimate the flux squared for Eddy Loss calculation.
                if q == 0:

                    # Max flux occurs at
                    # - innermost radius
                    # - at x (and phase) where cos(kx) is 1
                    # - at y on either edge
                    # sinh(x)**2 + cosh(x)**2 = cosh(2x)
                    #B_sq_max_est = Bterm**2 * np.cosh(2.0 * k * y[0])

                    # Simpler estimation of max flux from the paper 2.
                    B_sq_max_est = (Bterm * np.cosh(0)) ** 2

                # Torque for each coil at radius r.
                T_coil[q] = (r_q * np.sum(F))

            # Torque from each radii.
            T_total[z] = np.sum(T_coil) * npole

        # Power at Rpm. (W)
        P = np.sum(T_total) * omega / ntime

        # Eddy loss calculcation. (W)
        vol_cond = 2.0 * (RF - R0) * Awires * nphase * npole
        Pe = 0.5 * (np.pi * npole * RPM/60.0 * 2.0 * rwire)**2 * vol_cond / resistivity * B_sq_max_est
        PPe = 0.5 * nphase * npole * (np.pi * npole * RPM / 60.0 * 2.0 * rwire)**2 * (RF - R0) * Awires * B_sq_max_est / resistivity

        efficiency = (P - PR - Pe) / P

        # Mass sum. (kg)
        M_stator = wire_length * Awires * self.rho_stator * nphase * 2
        M = M_stator + M_magnet

        # Power Density (converted to kW/kg)
        power_density = P/M * 0.001

        # Convert all outputs to kg, where applicable
        outputs['power_ideal'] = P * 0.001
        outputs['power_density'] = power_density
        outputs['resistive_loss'] = PR * 0.001
        outputs['eddy_current_loss'] = Pe * 0.001
        outputs['efficiency'] = efficiency
        outputs['rotor_mass'] = M_magnet
        outputs['stator_mass'] = M_stator

    def compute_partials(self, inputs, partials):
        R0 = inputs['inner_radius'][0]
        RF = inputs['outer_radius'][0]
        xm = inputs['magnet_width'][0]
        ym = inputs['magnet_depth'][0]
        yw = inputs['coil_thickness'][0]
        ag = inputs['air_gap'][0]
        Ipeak = inputs['Ipeak'][0]
        RPM = inputs['RPM'][0]
        resistivity = inputs['resistivity'][0]

        nm = self.options['num_magnets']
        npole = self.options['num_poles']
        nphase = self.options['num_phases']
        xws_norm = self.options['coil_locations']

        minwall = self.minwall
        cfill = self.cfill
        Br = self.Br
        rwire = self.rwire
        rho_mag = self.rho_mag
        rho_stator = self.rho_stator

        ntime = self.ntime
        nr = self.nr

        # Back calculate from thickness of coil
        yg = 0.5 * yw + ag
        dyg_dyw = 0.5
        dyg_dag = 1.0

        # Area available for wire (I-beam area)
        A = (2.0*R0*pi/(nphase*npole) - minwall) * yw * 0.5
        dA_dR0 = 2.0 * pi / (nphase*npole) * yw * 0.5
        dA_dyw = (2.0 * R0 * pi / (nphase*npole) - minwall) * 0.5

        # Width of coil
        xw = A / yw
        dxw_dR0 = dA_dR0 / yw
        dxw_dyw = dA_dyw / yw - A / yw**2

        # Area of all wires in a single coil.
        Awires = A * cfill
        dAwires_dR0 = dA_dR0 * cfill
        dAwires_dyw = dA_dyw * cfill

        # Estimate Minimum Length of all Wire
        if self.options['overlap']:
            wire_length = 2.0*(RF-R0)*npole + pi*(RF + R0)
            dwl_dRF = 2.0 * npole + pi
            dwl_dR0 = -2.0 * npole + pi
        else:
            wire_length = 2.0*(RF-R0)*npole + 2.0*pi*(RF + R0)/nphase
            dwl_dRF = 2.0 * npole + 2.0 * pi / nphase
            dwl_dR0 = -2.0 * npole + 2.0 * pi / nphase

        # Discretize in radial direction
        dr = (RF - R0) / nr
        r = R0 + 0.5*dr + dr*np.arange(nr)
        dr_dRF = 0.5/nr + np.arange(nr)/nr
        dr_dR0 = 1.0 - 0.5/nr - np.arange(nr)/nr

        # Discretize in y: just need start and end
        y = np.array([-yg + ag, yg - ag])
        dy_yw = np.array([-dyg_dyw, dyg_dyw])
        dy_ag = np.array([-dyg_dag + 1, dyg_dag - 1])

        # Discretize in time
        omega = RPM / 30. * pi
        domega_drpm = pi / 30.0
        if ntime > 1:
            t_end = 2.0 * pi / (npole * omega)
            dt = t_end / (ntime-1)
            t = dt*np.arange(ntime)
            dt_drpm = -2.0 * pi * domega_drpm * np.arange(ntime) / (npole * (ntime-1) * omega**2)
        else:
            t = np.array([0.0])
            dt_drpm = np.array([0.0])

        # Magnet mass
        M_magnet = rho_mag * xm * ym * (RF - R0) * npole * nm * 2
        dMmag_dRF = rho_mag * xm * ym * npole * nm * 2
        dMmag_dR0 = -rho_mag * xm * ym * npole * nm * 2
        dMmag_dym = rho_mag * xm * (RF - R0) * npole * nm * 2
        dMmag_dxm = rho_mag * ym * (RF - R0) * npole * nm * 2

        # Calculate resistance.
        phase_R = wire_length * resistivity / (A * cfill)
        PR = nphase * 0.5 * phase_R * (Ipeak)**2
        dPR_dIpk = nphase * phase_R * Ipeak
        dPR_dRF = nphase * 0.5 * (Ipeak)**2 * dwl_dRF * resistivity / (A * cfill)
        dPR_dR0 = nphase * 0.5 * (Ipeak)**2 * resistivity * (dwl_dR0 / (A * cfill) - wire_length * dA_dR0 / (cfill * A**2))
        dPR_dyw = -nphase * 0.5 * (Ipeak)**2 * wire_length * resistivity * dA_dyw / (cfill * A**2)
        dPR_dresist = nphase * 0.5* (Ipeak)**2 * wire_length / (A * cfill)

        T_coil = np.empty((nr, ), dtype=inputs._data.dtype)
        T_total = np.empty((ntime, ), dtype=inputs._data.dtype)

        T_coil_dRF = 0
        T_coil_dR0 = 0
        T_coil_dym = 0
        T_coil_dxm = 0
        T_coil_dyw = 0
        T_coil_dag = 0
        T_coil_dIpk = 0
        T_coil_drpm = 0

        # Integrate over time.
        for z in range(ntime):
            t_z = t[z]

            # Define Coil Currents in each coil direction
            I = Ipeak * np.cos(npole * omega * t_z - (2.0 * pi / nphase) * np.arange(nphase))
            I = np.append(I, -I)
            J = I / Awires
            dJ_dR0 = -I / Awires**2 * dAwires_dR0
            dJ_dyw = -I / Awires**2 * dAwires_dyw
            dJ_dIpk = J / Ipeak
            dJ_drpm = -Ipeak * np.sin(npole * omega * t_z - (2.0 * pi / nphase) * np.arange(nphase)) * \
                      npole * (t_z * domega_drpm + omega * dt_drpm) / Awires
            dJ_drpm = np.append(dJ_drpm, -dJ_drpm)

            # Adjust position of rotor relative to stator at given point in time.
            x_adjust = omega * r * t_z
            dxadj_dRF = omega * dr_dRF * t_z
            dxadj_dR0 = omega * dr_dR0 * t_z
            dxadj_drpm = r * (domega_drpm * t_z + omega * dt_drpm[z])

            # m length of one pole pair at radius.
            xp = 2.0 * pi * r / npole
            dxp_dRF = dr_dRF * 2.0 * pi / npole
            dxp_dR0 = dr_dR0 * 2.0 * pi / npole

            # Percent of material that is magnet in x direction.
            e = (nm * xm) / xp
            de_dRF = -(nm * xm) / xp**2 * dxp_dRF
            de_dR0 = -(nm * xm) / xp**2 * dxp_dR0
            de_dxm = nm / xp

            for q in range(nr):
                r_q = r[q]

                # Define Coil Currents and x start points.
                xws = xws_norm * xp[q] + x_adjust[q]
                dxws_dRF = xws_norm * dxp_dRF[q] + dxadj_dRF[q]
                dxws_dR0 = xws_norm * dxp_dR0[q] + dxadj_dR0[q]
                dxws_drpm = dxadj_drpm[q]

                k = 2.0 * pi / xp[q]
                dk_dRF = -2.0 * pi / xp[q]**2 * dxp_dRF[q]
                dk_dR0 = -2.0 * pi / xp[q]**2 * dxp_dR0[q]

                Bterm = 2.0 * Br * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e[q]*pi/nm) * nm/pi
                dBt_dRF = 2.0 * Br * (-yg) * dk_dRF * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e[q]*pi/nm) * nm/pi + \
                          2.0 * Br * np.exp(-k*yg) * ym * dk_dRF * np.exp(-k*ym) * np.sin(e[q]*pi/nm) * nm/pi  + \
                          2.0 * Br * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.cos(e[q]*pi/nm) * de_dRF[q]
                dBt_dR0 = 2.0 * Br * (-yg) * dk_dR0 * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e[q]*pi/nm) * nm/pi + \
                          2.0 * Br * np.exp(-k*yg) * ym * dk_dR0 * np.exp(-k*ym) * np.sin(e[q]*pi/nm) * nm/pi  + \
                          2.0 * Br * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.cos(e[q]*pi/nm) * de_dR0[q]
                dBt_dyw = 2.0 * Br * (-k) * dyg_dyw * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e[q]*pi/nm) * nm/pi
                dBt_dag = 2.0 * Br * (-k) * dyg_dag * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.sin(e[q]*pi/nm) * nm/pi
                dBt_dym = 2.0 * Br * np.exp(-k*yg) * k * np.exp(-k*ym) * np.sin(e[q]*pi/nm) * nm/pi
                dBt_dxm = 2.0 * Br * np.exp(-k*yg) * (1.0 - np.exp(-k*ym)) * np.cos(e[q]*pi/nm) * de_dxm[q]

                sinh_ky = np.sinh(k*y)
                dsnh_dyw = np.cosh(k*y) * k * dy_yw
                dsnh_dag = np.cosh(k*y) * k * dy_ag
                dsnh_dRF = np.cosh(k*y) * y * dk_dRF
                dsnh_dR0 = np.cosh(k*y) * y * dk_dR0

                # Force is calculated for all coils, current always symmetric about center
                fact1 = (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (sinh_ky[-1] - sinh_ky[0]) / k**2
                By = Bterm * fact1
                dBy_dRF = dBt_dRF * fact1 + Bterm * (
                          (np.cos(k*(xws+xw)) * (xws+xw) - np.cos(k*xws) * xws) * dk_dRF * (sinh_ky[-1] - sinh_ky[0]) / k**2 + \
                          (np.cos(k*(xws+xw)) - np.cos(k*xws)) * k * dxws_dRF * (sinh_ky[-1] - sinh_ky[0]) / k**2 + \
                          (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (dsnh_dRF[-1] - dsnh_dRF[0]) / k**2  + \
                          (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (sinh_ky[-1] - sinh_ky[0]) * -2.0 / k**3 * dk_dRF
                )
                dBy_dR0 = dBt_dR0 * fact1 + Bterm * (
                          (np.cos(k*(xws+xw)) * (xws+xw) - np.cos(k*xws) * xws) * dk_dR0 * (sinh_ky[-1] - sinh_ky[0]) / k**2 + \
                          (np.cos(k*(xws+xw)) * (dxws_dR0 + dxw_dR0) - np.cos(k*xws) * dxws_dR0) * k * (sinh_ky[-1] - sinh_ky[0]) / k**2 + \
                          (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (dsnh_dR0[-1] - dsnh_dR0[0]) / k**2 + \
                          (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (sinh_ky[-1] - sinh_ky[0]) * -2.0 / k**3 * dk_dR0
                )
                dBy_dyw = dBt_dyw * fact1 + Bterm * (
                    np.cos(k*(xws+xw)) * dxw_dyw * k * (sinh_ky[-1] - sinh_ky[0]) / k**2 + \
                    (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (dsnh_dyw[-1] - dsnh_dyw[0]) / k**2
                )
                dBy_dag = dBt_dag * fact1 + Bterm * (
                    (np.sin(k*(xws+xw)) - np.sin(k*xws)) * (dsnh_dag[-1] - dsnh_dag[0]) / k**2
                )
                dBy_dym = dBt_dym * fact1
                dBy_dxm = dBt_dxm * fact1
                dBy_drpm = Bterm * (np.cos(k*(xws+xw)) - np.cos(k*xws)) * k * dxws_drpm * (sinh_ky[-1] - sinh_ky[0]) / k**2


                # Flux to Force
                F = J * By * dr
                dF_dRF = J * dBy_dRF * dr + J * By / nr
                dF_dR0 = (J * dBy_dR0 + dJ_dR0 * By) * dr - J * By / nr
                dF_dym = J * dr * dBy_dym
                dF_dxm = J * dr * dBy_dxm
                dF_dyw = (J * dBy_dyw + dJ_dyw * By) * dr
                dF_dag = J * dr * dBy_dag
                dF_dIpk = dJ_dIpk * By * dr
                dF_drpm = (J * dBy_drpm + dJ_drpm * By) * dr

                if q == 0:
                    #B_sq_max_est = Bterm**2 * np.cosh(2.0 * k * y[0])
                    B_sq_max_est = (Bterm * np.cosh(0)) ** 2

                    Dbsqmax_dRF = 2.0 * Bterm * dBt_dRF * np.cosh(0)
                    Dbsqmax_dR0 = 2.0 * Bterm * dBt_dR0 * np.cosh(0)
                    Dbsqmax_dyw = 2.0 * Bterm * dBt_dyw * np.cosh(0)
                    Dbsqmax_dag = 2.0 * Bterm * dBt_dag * np.cosh(0)
                    Dbsqmax_dym = 2.0 * Bterm * dBt_dym * np.cosh(0)
                    Dbsqmax_dxm = 2.0 * Bterm * dBt_dxm * np.cosh(0)

                # Torque for each coil at radius r.
                T_coil[q] = (r_q * np.sum(F))
                T_coil_dRF += r_q * np.sum(dF_dRF) + dr_dRF[q] * np.sum(F)
                T_coil_dR0 += r_q * np.sum(dF_dR0) + dr_dR0[q] * np.sum(F)
                T_coil_dym += r_q * np.sum(dF_dym)
                T_coil_dxm += r_q * np.sum(dF_dxm)
                T_coil_dyw += r_q * np.sum(dF_dyw)
                T_coil_dag += r_q * np.sum(dF_dag)
                T_coil_dIpk += r_q * np.sum(dF_dIpk)
                T_coil_drpm += r_q * np.sum(dF_drpm)

            # Torque from each radii.
            T_total[z] = np.sum(T_coil) * npole

        # Power at Rpm. (W)
        P = np.sum(T_total) * omega / ntime
        dP_dRF = T_coil_dRF * npole * omega / ntime
        dP_dR0 = T_coil_dR0 * npole * omega / ntime
        dP_dym = T_coil_dym * npole * omega / ntime
        dP_dxm = T_coil_dxm * npole * omega / ntime
        dP_dyw = T_coil_dyw * npole * omega / ntime
        dP_dag = T_coil_dag * npole * omega / ntime
        dP_dIpk = T_coil_dIpk * npole * omega / ntime
        dP_drpm = (T_coil_drpm * omega + np.sum(T_total) * domega_drpm) / ntime

        # Eddy loss calculcation. (W)
        vol_cond = 2.0 * (RF - R0) * Awires * nphase * npole
        fact = 0.5 * (np.pi * npole * RPM/60.0 * 2.0 * rwire)**2
        Pe = fact * vol_cond / resistivity * B_sq_max_est
        dPe_dRF = fact / resistivity * (vol_cond * Dbsqmax_dRF + \
                                        2.0 * Awires * nphase * npole * B_sq_max_est)
        dPe_dR0 = fact / resistivity * (vol_cond * Dbsqmax_dR0 + \
                                        (-Awires + (RF - R0) * dAwires_dR0) * 2.0 * nphase * npole * B_sq_max_est)
        dPe_dyw = fact / resistivity * (vol_cond * Dbsqmax_dyw + \
                                        (RF - R0) * dAwires_dyw * 2.0 * nphase * npole * B_sq_max_est)
        dPe_dag = fact * vol_cond / resistivity * Dbsqmax_dag
        dPe_dym = fact * vol_cond / resistivity * Dbsqmax_dym
        dPe_dxm = fact * vol_cond / resistivity * Dbsqmax_dxm
        dPe_dresist = - fact * vol_cond * B_sq_max_est / resistivity**2
        dPe_drpm = (np.pi * npole * rwire / 30.0)**2 * RPM * B_sq_max_est * vol_cond / resistivity

        # Mass sum. (kg)
        M_stator = wire_length * Awires * self.rho_stator * nphase * 2
        dMstat_dRF = dwl_dRF * Awires * rho_stator * nphase * 2
        dMstat_dR0 = (dwl_dR0 * Awires + wire_length * dA_dR0 * cfill) * rho_stator * nphase * 2
        dMstat_dyw = wire_length * dA_dyw * cfill * rho_stator * nphase * 2

        M = M_stator + M_magnet

        partials['power_ideal', 'outer_radius'] = dP_dRF * .001
        partials['power_ideal', 'inner_radius'] = dP_dR0 * .001
        partials['power_ideal', 'magnet_depth'] = dP_dym * .001
        partials['power_ideal', 'magnet_width'] = dP_dxm * .001
        partials['power_ideal', 'coil_thickness'] = dP_dyw * .001
        partials['power_ideal', 'air_gap'] = dP_dag * .001
        partials['power_ideal', 'Ipeak'] = dP_dIpk * .001
        partials['power_ideal', 'RPM'] = dP_drpm * .001

        partials['rotor_mass', 'outer_radius'] = dMmag_dRF
        partials['rotor_mass', 'inner_radius'] = dMmag_dR0
        partials['rotor_mass', 'magnet_depth'] = dMmag_dym
        partials['rotor_mass', 'magnet_width'] = dMmag_dxm

        partials['stator_mass', 'outer_radius'] = dMstat_dRF
        partials['stator_mass', 'inner_radius'] = dMstat_dR0
        partials['stator_mass', 'coil_thickness'] = dMstat_dyw

        partials['power_density', 'outer_radius'] = (dP_dRF/M - P/M**2 * (dMmag_dRF + dMstat_dRF)) * .001
        partials['power_density', 'inner_radius'] = (dP_dR0/M - P/M**2 * (dMmag_dR0 + dMstat_dR0)) * .001
        partials['power_density', 'magnet_depth'] = (dP_dym/M - P/M**2 * dMmag_dym) * .001
        partials['power_density', 'magnet_width'] = (dP_dxm/M - P/M**2 * dMmag_dxm) * .001
        partials['power_density', 'coil_thickness'] = (dP_dyw/M - P/M**2 * dMstat_dyw) * .001
        partials['power_density', 'air_gap'] = dP_dag/M * .001
        partials['power_density', 'Ipeak'] = dP_dIpk/M * .001
        partials['power_density', 'RPM'] = dP_drpm/M * .001

        partials['resistive_loss', 'Ipeak'] = dPR_dIpk * .001
        partials['resistive_loss', 'outer_radius'] = dPR_dRF * .001
        partials['resistive_loss', 'inner_radius'] = dPR_dR0 * .001
        partials['resistive_loss', 'coil_thickness'] = dPR_dyw * .001
        partials['resistive_loss', 'resistivity'] = dPR_dresist * .001

        partials['eddy_current_loss', 'outer_radius'] = dPe_dRF * .001
        partials['eddy_current_loss', 'inner_radius'] = dPe_dR0 * .001
        partials['eddy_current_loss', 'magnet_depth'] = dPe_dym * .001
        partials['eddy_current_loss', 'magnet_width'] = dPe_dxm * .001
        partials['eddy_current_loss', 'coil_thickness'] = dPe_dyw * .001
        partials['eddy_current_loss', 'air_gap'] = dPe_dag * .001
        partials['eddy_current_loss', 'RPM'] = dPe_drpm * .001
        partials['eddy_current_loss', 'resistivity'] = dPe_dresist * .001

        partials['efficiency', 'outer_radius'] = (dP_dRF - dPR_dRF - dPe_dRF) / P - (P - PR - Pe) * dP_dRF / P**2
        partials['efficiency', 'inner_radius'] = (dP_dR0 - dPR_dR0 - dPe_dR0) / P - (P - PR - Pe) * dP_dR0 / P**2
        partials['efficiency', 'magnet_depth'] = (dP_dym - dPe_dym) / P - (P - PR - Pe) * dP_dym / P**2
        partials['efficiency', 'magnet_width'] = (dP_dxm - dPe_dxm) / P - (P - PR - Pe) * dP_dxm / P**2
        partials['efficiency', 'coil_thickness'] = (dP_dyw - dPR_dyw - dPe_dyw) / P - (P - PR - Pe) * dP_dyw / P**2
        partials['efficiency', 'air_gap'] = (dP_dag - dPe_dag) / P - (P - PR - Pe) * dP_dag / P**2
        partials['efficiency', 'Ipeak'] = (dP_dIpk  - dPR_dIpk )/ P - (P - PR - Pe) * dP_dIpk / P**2
        partials['efficiency', 'RPM'] = (dP_drpm - dPe_drpm) / P - (P - PR - Pe) * dP_drpm / P**2
        partials['efficiency', 'resistivity'] = (-dPR_dresist - dPe_dresist) / P


class DoubleHalbachMotorThermalComp(ExplicitComponent):
    """
    Component that models the thermal properties of the Halbach Array motor, computing the stator
    temperature and the coil resistivity assuming a copper conductor.

    The thermal equations come from convective flow over a flat plate.

    This is an experimental component. Equations have been published, but the application is for a
    specific configuration of motor. For the X57, this would not be an adequate method of cooling.
    """

    def __init__(self, **kwargs):
        """
        Initialize attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(DoubleHalbachMotorThermalComp, self).__init__(**kwargs)
        self.cite = CITATION2

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

        self.add_input('Cp_air', val=1.000, units='kJ/(kg*degK)',
                        desc='Specific heat capactity at constant pressure for air.')

        self.add_input('k_air', val=2.4e-5, units='kW/(m*degK)',
                       desc='Thermal conductivity of air.')

        self.add_input('T_air', val=272.0, units='degK',
                       desc='Temperature for air.')

        self.add_input('V_air', val=70.0, units='m/s',
                        desc='Velocity of air.')

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

        # (kg/m**3) * (m/s) / (N*s/m**2)
        # kg / (N * s**2)
        # 1/m <-- Bad units for Re
        # Seems to be assumption of unit length.
        Re = rho * V / mu

        # (N*s/m**2) * kJ/(kg*degK) / kW/(m*degK)
        # kg/(s*m) * kN*m/(kg*degK) * m*degK/(kN*m/s)
        # units balance
        Pr = mu * Cp / k

        if Pr < 0.6:
            msg = "Warning: Prandtl number of %f is not within supported range (Pr > 0.6)" % Pr
            raise RuntimeWarning(msg)

        Hconv = 0.664 * k * Pr**(1.0/3.0) * np.sqrt(Re)
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
        Hconv = 0.664 * k * Pr**(1.0/3.0) * np.sqrt(Re)

        dHconv_drho = 0.664 * k * Pr**(1.0/3.0) * 0.5 / np.sqrt(Re) * V / mu
        dHconv_dmu = 0.664 * k * (Pr**(-2.0/3.0) / 3.0 * Cp / k * np.sqrt(Re) +
                                  Pr**(1.0/3.0) * 0.5 / np.sqrt(Re) * (-rho * V / mu**2))
        dHconv_dCp = 0.664 * k * Pr**(-2.0/3.0) / 3.0 * mu / k * np.sqrt(Re)
        dHconv_dV = 0.664 * k * Pr**(1.0/3.0) * 0.5 / np.sqrt(Re) * rho / mu
        dHconv_dk = 0.664 * np.sqrt(Re) * (Pr**(1.0/3.0) + k * Pr**(-2.0/3.0) / 3.0 * -mu * Cp / k**2)

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


class BearingLosses(ExplicitComponent):
    """
    Component that models the bearing losses for the motor used to power a propulsor.

    Can be optionally used for "high lift" configuration by setting the option.

    Requires the net thrust produced by the motor/fan combination.
    """

    def __init__(self, **kwargs):
        """
        Initialize attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(BearingLosses, self).__init__(**kwargs)
        self.cite = CITATION2

    def initialize(self):
        """
        Declare options accessable via the self.options dictionary.
        """
        self.options.declare('high_lift', False,
                             desc='Set to True to when used to provide lift.')

        self.options.declare('mu', .0024,
                             desc='Friction Coefficient.')

        self.options.declare('bore', .025,
                             desc='Bearing bore diameter. (units m)')

    def setup(self):
        """
        Declare input and output for this component, and declare derivatives.
        """

        # Inputs
        self.add_input('net_thrust', val=0.0, units='N',
                       desc='Net axial force on shaft.')

        self.add_input('RPM', val=8600., units='rpm',
                       desc='Motor rotational speed.')

        self.add_input('rotor_mass', val=6.0, units='kg',
                       desc='Mass of the rotor.')

        self.add_input('fan_mass', val=7.7, units='kg',
                       desc='Mass of the fan.')

        # Outputs
        self.add_output('bearing_loss', units='kW',
                        desc='Losses due to bearings.')

        # Derivatives
        # -----------
        self.declare_partials(of='*', wrt='*')

        self.g = 9.81

    def compute(self, inputs, outputs):
        """
        Compute thermal properties.
        """
        options = self.options

        if options['high_lift']:
            force = inputs['net_thrust'] + self.g * (inputs['rotor_mass'] + inputs['fan_mass'])
        else:
            force = inputs['net_thrust']

        w = inputs['RPM'] * pi / 30.0

        # Convert from W to KW
        outputs['bearing_loss'] = 0.5 * options['mu'] * options['bore'] * force * w * .001

    def compute_partials(self, inputs, partials):
        options = self.options

        bore = options['bore']
        mu = options['mu']

        if options['high_lift']:
            m_rotor = inputs['rotor_mass']
            m_fan = inputs['fan_mass']

            g = self.g
            force = inputs['net_thrust'] + g * (m_rotor + m_fan)
        else:
            force = inputs['net_thrust']

        w = inputs['RPM'] * pi / 30.0

        partials['bearing_loss', 'RPM'] = 0.5 * mu * bore * force * pi / 30.0 * .001
        partials['bearing_loss', 'net_thrust'] = 0.5 * mu * bore * w * .001

        if options['high_lift']:
            partials['bearing_loss', 'rotor_mass'] = 0.5 * mu * bore * w * g * .001
            partials['bearing_loss', 'fan_mass'] = 0.5 * mu * bore * w * g * .001


class WindageLosses(ExplicitComponent):
    """
    Component that models the windage losses for the motor in rim drive configuration.
    """

    def __init__(self, **kwargs):
        """
        Initialize attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(WindageLosses, self).__init__(**kwargs)
        self.cite = CITATION2

    def setup(self):
        """
        Declare input and output for this component, and declare derivatives.
        """

        # Inputs
        self.add_input('RPM', val=8600., units='rpm',
                       desc='Motor rotational speed.')

        self.add_input('inner_radius', val=6.125 * 0.0254, units='m',
                           desc='Rotor inner radius.')

        self.add_input('outer_radius', val=6.875 * 0.0254, units='m',
                           desc='Rotor outer radius.')

        self.add_input('rho_air', val=1.0, units='kg/m**3',
                        desc='Density for air.')

        self.add_input('mu_air', val=1.7e-5, units='N*s/m**2',
                        desc='Dynamic viscosity for air.')

        self.add_input('air_gap', .001, units='m',
                       desc = 'Air gap between rotor and stator.')

        self.add_input('hoop_thickness', .001, units='m',
                       desc='Thickness of carbon fiber restraining hoop.')

        # Outputs
        self.add_output('windage_loss', units='kW',
                        desc='Losses due to bearings.')

        # Derivatives
        # -----------
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        """
        Compute thermal properties.
        """
        R0 = inputs['inner_radius']
        RF = inputs['outer_radius']
        rho = inputs['rho_air']
        mu = inputs['mu_air']
        ag = inputs['air_gap']
        tk = inputs['hoop_thickness']

        w = inputs['RPM'] * pi / 30.0

        RFW = RF + tk
        Re = rho * w * RFW**2 / mu
        Cf = .08 / ((ag/R0)**.167 * Re**.25)

        # Convert from W to KW
        outputs['windage_loss'] = 0.5 * Cf * rho * w**3 * (RFW**5 - R0**5) * .001

    def compute_partials(self, inputs, partials):
        R0 = inputs['inner_radius']
        RF = inputs['outer_radius']
        rho = inputs['rho_air']
        mu = inputs['mu_air']
        ag = inputs['air_gap']
        tk = inputs['hoop_thickness']

        w = inputs['RPM'] * pi / 30.0

        RFW = RF + tk
        Re = rho * w * RFW**2 / mu
        Cf = .08 / ((ag/R0)**.167 * Re**.25)

        r5term = (RFW**5 - R0**5)

        dCf_dRe = -.25 * .08 / ((ag/R0)**.167 * Re**1.25)

        dCf_dag = -.167 * .08 / ((ag/R0)**1.167 * Re**.25 * R0)
        partials['windage_loss', 'air_gap'] = 0.5 * dCf_dag * rho * w**3 * r5term * .001

        dCf_drho = dCf_dRe * w * RFW**2 / mu
        partials['windage_loss', 'rho_air'] = 0.5 * w**3 * r5term * .001 * (dCf_drho * rho + Cf)

        dCf_dmu = dCf_dRe * (-rho * w * RFW**2 / mu**2)
        partials['windage_loss', 'mu_air'] = 0.5 * dCf_dmu * rho * w**3 * r5term * .001

        dCf_dtk = dCf_dRe * (2.0 * rho * w * RFW / mu)
        partials['windage_loss', 'hoop_thickness'] = 0.5 * rho * w**3  * .001 * (dCf_dtk * r5term +
                                                                                 Cf * 5.0 * RFW**4)

        partials['windage_loss', 'outer_radius'] = partials['windage_loss', 'hoop_thickness']

        dCf_dR0 = .167 * .08 * ag / ((ag/R0)**1.167 * Re**.25 * R0**2)
        partials['windage_loss', 'inner_radius'] = 0.5 * rho * w**3  * .001 * (dCf_dR0 * r5term -
                                                                               Cf * 5.0 * R0**4)
        dCf_dw = dCf_dRe * (rho * RFW**2 / mu)
        partials['windage_loss', 'RPM'] = 0.5 * rho * r5term * .001 * pi / 30.0 * (dCf_dw * w**3 +
                                                                                   Cf * 3.0 * w**2)

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