""" Some basic, but not exhaustive tests. Mostly testing the derivatives."""

import unittest

from openmdao.api import Problem, NonlinearBlockGS
from openmdao.utils.assert_utils import assert_check_partials

from double_halbach_motor import DoubleHalbachMotorComp, DoubleHalbachMotorThermalComp


class TestHalbachMotor(unittest.TestCase):

    def test_thermal_derivatives(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('comp', DoubleHalbachMotorThermalComp())

        prob.setup(force_alloc_complex=True)

        prob['comp.resistive_loss'] = 0.11
        prob['comp.eddy_current_loss'] = 0.27

        prob.run_model()

        J = prob.check_partials(method='cs')

        assert_check_partials(J, atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
