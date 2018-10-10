"""
OpenMDAO component modeling a double Halbach array permanent-magnet ironless axial-flux motor.

Based on work by Kirsten P. Duffy.
OpenMDAO Component by Kenneth T. Moore
"""
from __future__ import division, print_function
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
        self.lm1 = 10.0       # Magnet length (radial)
        self.npole = 24       # Number of magnetic pole pairs
        self.nm = 4           # Number of magnets in a pole pair
        self.Br = 1.0         # Permanent magnet remanence flux (T)
        self.md = 7300.       # Magnet material mass density (kg/m**3)

        # Stator Material and Physical Constant Parameters
        self.ag = .001              # Air gap between rotor and stator. (m)
        self.nphase = 3             # Number of Phases
        self.cd = 8960.             # Density of stator material (Cu) (kg/m**3)
        self.Imax = 6.0             # Max current for 1 wire (Amp)
        self.R = 34.1               # Wire resistance (ohms/Km)
        self.minwall = .1 * .0254   # Min thickness of wire cores (tooth) at inner r (m)
        self.cfill = .5             # Copper fill percentage

        rwire = .0004
        self.Awire = pi*(rwire)**2  # Square volume taken up by 1 wire (m**2)

        # Discretization
        self.nr = 100
        self.nx = 10
        self.ny = 9
        self.ntime = 10

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

        self.add_output('resistive_loss',
                        desc='Resistive loss as a percentage of total power.')

        self.add_output('eddy_current_loss',
                        desc='Eddy current loss as a percentage of total power.')

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
        npole = self.npole
        nphase = self.nphase
        cfill = self.cfill
        Awire = self.Awire

        # Area available for wire (I-beam area)
        A = (2.0*R0*pi/(nphase*npole) - minwall) * (2.0*(yg-ag)) * 0.5

        # Number of wires per phase
        nw = np.floor(A * cfill/Awire)

        # Estimate Minimum Length of 1 Wire
        wire_length = nw*(2.0*(RF-R0) + (2.0*RF*pi + 2*R0*pi)/(npole*nphase))

        # Discretize in radial direction
        dr = (RF - R0) / self.nr
        r = np.arange((R0 + 0.5*dr), (RF - 0.5*dr), dr)

        # Discretize in y
        dy = 2.0 * (yg - ag) / self.ny
        y = np.arange((-yg + ag), (yg - ag), dy)

        print('zz')


if __name__ == "__main__":

    from openmdao.api import Problem, IndepVarComp

    prob = Problem()
    prob.model.add_subsystem('comp', DoubleHalbachMotorComp())

    prob.setup()
    prob.run()
