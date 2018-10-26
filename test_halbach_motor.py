""" Some basic, but not exhaustive tests. Mostly testing the derivatives."""

import unittest

from openmdao.api import Problem, NonlinearBlockGS
from openmdao.utils.assert_utils import assert_check_partials

from double_halbach_motor import DoubleHalbachMotorComp, DoubleHalbachMotorThermalComp, \
     BearingLosses, WindageLosses


class TestHalbachMotor(unittest.TestCase):

    def test_thermal_derivatives(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('comp', DoubleHalbachMotorThermalComp())

        prob.setup(force_alloc_complex=True)

        # kW of loss
        prob['comp.resistive_loss'] = 1.5
        prob['comp.eddy_current_loss'] = 6.3

        prob.run_model()

        J = prob.check_partials(method='cs')

        assert_check_partials(J, atol=1e-7, rtol=1e-7)

    def test_motor_derivatives(self):
        prob = Problem()
        model = prob.model

        comp = model.add_subsystem('comp', DoubleHalbachMotorComp())

        prob.setup(force_alloc_complex=True)

        comp.ntime = 3
        comp.nr = 4

        prob['comp.magnet_width'] = .0042
        prob['comp.magnet_depth'] = .0047

        prob.run_model()

        J = prob.check_partials(method='cs', compact_print=True)

        assert_check_partials(J, atol=1e-7, rtol=1e-7)

    def test_bearing_loss_derivatives(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('comp', BearingLosses(high_lift=True))

        prob.setup(force_alloc_complex=True)

        prob['comp.RPM'] = 3500
        prob['comp.net_thrust'] = 1.0

        prob.run_model()

        J = prob.check_partials(method='cs')

        assert_check_partials(J, atol=1e-7, rtol=1e-7)

    def test_windage_loss_derivatives(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('comp', WindageLosses())

        prob.setup(force_alloc_complex=True)

        prob['comp.rho_air'] = 0.7

        prob.run_model()

        J = prob.check_partials(method='cs')

        assert_check_partials(J, atol=1e-7, rtol=1e-7)

if __name__ == "__main__":
    unittest.main()
