"""
OpenMDAO component modeling a double Halbach array permanent-magnet ironless axial-flux motor.

Based on work by Thomas Tallerico, Jeff Chin, and Zachery Cameron.
OpenMDAO Component by Kenneth T. Moore
"""
from __future__ import division, print_function

import numpy as np

from openmdao.api import Group, IndepVarComp, ExecComp, NonlinearBlockGS

from double_halbach_motor import DoubleHalbachMotorComp, DoubleHalbachMotorThermalComp, BearingLosses, \
     WindageLosses

CITATION = """@PROCEEDINGS{
              Tallerico2018,
              AUTHOR = "T. Tallerico, J. Chin, Z. Cameron",
              TITLE = "Optimization of an Air Core Dual Halbach Array Axial Flux Rim Drive for "
                      "Electric Aircraft",
              BOOKTITLE = "AIAA Aviation Forum",
              PUBLISHER = "AIAA",
              YEAR = "2018",
              MONTH = "JUNE",
              }"""


class MotorGroup(Group):

    def initialize(self):
        """
        Declare options accessable via the self.options dictionary.
        """
        self.options.declare('thermal', False,
                             desc='Set to True to include experimental thermal model.')


    def setup(self):

        dv = self.add_subsystem('dv', IndepVarComp())
        dv.add_output('inner_radius', val=6.125 * 0.0254, units='m',
                      desc='Rotor inner radius.')
        dv.add_output('outer_radius', val=6.875 * 0.0254, units='m',
                      desc='Rotor outer radius.')
        dv.add_output('RPM', val=8600., units='rpm',
                      desc='Motor rotational speed.')
        dv.add_output('magnet_width', val=.005, units='m',
                      desc='Magnet width (tangential).')
        dv.add_output('magnet_depth', val=.005, units='m',
                      desc='Magnet depth (axial).')
        dv.add_output('coil_thickness', val=.0066 , units='m',
                      desc='Coil Thickness.')
        dv.add_output('Ipeak', val=6.0 , units='A',
                      desc='Motor peak current.')
        dv.add_output('air_gap', .001, units='m',
                      desc = 'Air gap between rotor and stator.')

        self.add_subsystem('motor', DoubleHalbachMotorComp(overlap=True))

        for name in ['inner_radius', 'outer_radius', 'RPM', 'magnet_depth', 'magnet_width', 'coil_thickness',
                     'Ipeak', 'air_gap']:
            self.connect('dv.' + name, 'motor.' + name)

        if self.options['thermal']:

            self.add_subsystem('thermal', DoubleHalbachMotorThermalComp())

            self.connect('dv.inner_radius', 'thermal.inner_radius')
            self.connect('dv.coil_thickness', 'thermal.coil_thickness')
            self.connect('motor.resistive_loss', 'thermal.resistive_loss')
            self.connect('thermal.resistivity', 'motor.resistivity')

            # Light coupling introduced by thermal model.
            self.nonlinear_solver = NonlinearBlockGS()

        self.add_subsystem('bearing', BearingLosses())

        self.connect('dv.RPM', 'bearing.RPM')
        self.connect('motor.rotor_mass', 'bearing.rotor_mass')

        self.add_subsystem('windage', WindageLosses())

        self.connect('dv.RPM', 'windage.RPM')
        self.connect('dv.inner_radius', 'windage.inner_radius')
        self.connect('dv.outer_radius', 'windage.outer_radius')

        # Summation of all losses to give net power.
        self.add_subsystem('sum_losses', ExecComp("net_power = power_ideal - resistive_loss - eddy_current_loss - bearing_loss - windage_loss",
                           power_ideal={'units' : 'kW'}, resistive_loss={'units' : 'kW'}, eddy_current_loss={'units' : 'kW'},
                           bearing_loss={'units' : 'kW'}, windage_loss={'units' : 'kW'}, net_power={'units' : 'kW'}))

        self.connect('motor.power_ideal', 'sum_losses.power_ideal')
        self.connect('motor.resistive_loss', 'sum_losses.resistive_loss')
        self.connect('motor.eddy_current_loss', 'sum_losses.eddy_current_loss')
        self.connect('bearing.bearing_loss', 'sum_losses.bearing_loss')
        self.connect('windage.windage_loss', 'sum_losses.windage_loss')


if __name__ == "__main__":

    from openmdao.api import Problem

    prob = Problem()
    prob.model = MotorGroup()

    prob.setup()

    prob.run()

    print('Power', prob['sum_losses.net_power'])
    print('power_ideal', prob['motor.power_ideal'])
    print('resistive_loss', prob['motor.resistive_loss'])
    print('eddy_current_loss', prob['motor.eddy_current_loss'])
    print('bearing_loss', prob['bearing.bearing_loss'])
    print('windage_loss', prob['windage.windage_loss'])
