"""
Reproduce the results presented in the Duffy paper using the OpenMDAO component
"""
from __future__ import division, print_function
from six.moves import range

import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, view_model

from double_halbach_motor import DoubleHalbachMotorComp

NX = 40


class MotorModel(Group):


    def setup(self):

        # Outer diameter is fixed (mm)
        od_fixed = 139.7

        # Convert to radius (m)
        or_fixed = od_fixed * 0.5 * .001

        # Build model

        indeps = self.add_subsystem('p', IndepVarComp(), promotes=['*'])
        indeps.add_output('ID_OD_ratio', 0.5)
        indeps.add_output('outer_radius', or_fixed, units='m')

        self.add_subsystem('calc_inner_radius', ExecComp(['inner_radius = ID_OD_ratio * outer_radius']),
                           promotes=['*'])

        self.add_subsystem('motor', DoubleHalbachMotorComp(), promotes=['*'])



prob = Problem(model=MotorModel())

prob.setup()
motor = prob.model.motor

# -------------------------
# Power vs OD/ID (Figure 5, 6, 7)
# -------------------------

desvar = np.linspace(0.45, 0.9, NX)
sidevar = [3, 10, 20, 30]
#sidevar = [30]
result1 = np.empty((NX, 4))
result2 = np.empty((NX, 4))
result3 = np.empty((NX, 4))

for k, val2 in enumerate(sidevar):
    for j, val in enumerate(desvar):

        # Paper calculates flux at average radius.
        motor.nr = 1

        prob['motor.RPM'] = 7200
        yg = prob['motor.yg'] = 0.5 * .003 + motor.ag
        npole = motor.npole = 16
        nm = motor.nm = 10
        minwall = motor.minwall = 0
        ag = motor.ag
        nphase = motor.nphase
        cfill = motor.cfill = 0.5

        # This is calculated from rwire, so don't set it.
        Awire = motor.Awire

        # Paper sets a constant current density. We can mimic this by varying the
        # max current.
        Rf = 139.7 * 0.5 * .001
        R0 = val * Rf
        Rmid = 0.5 * (Rf + R0)

        # Main difference: Duffy ignores cfill during calculation of current density.
        yw = 2.0 * (yg - ag)
        coil_area = yw * (2.0 * np.pi * Rmid)/(nphase * 2 * npole)

        Imax = val2 * coil_area * 1.0e6
        motor.Imax = Imax

        # Set Magnet width so that each design has maximum e percentage at the midpoint radius
        xp = 2.0 * np.pi * Rmid / npole
        xm = xp / nm

        if True or j == 0:
            prob['magnet_width'] = xm
            prob['magnet_depth'] = xm

        # ID/OD ratio is a design var.
        prob['ID_OD_ratio'] = val

        prob.run_model()

        result1[j, k] = prob['power']
        result3[j, k] = prob['power_density']
        result2[j, k] = prob['resistive_loss'] / prob['power'] * 100.0

        # print('Current Density', motor.current_density)


plt.figure(1)
plt.plot(desvar, result1[:, 0])
plt.plot(desvar, result1[:, 1])
plt.plot(desvar, result1[:, 2])
plt.plot(desvar, result1[:, 3])
plt.xlabel("Ratio of Motor ID to OD")
plt.ylabel('Motor Power (KW)')
plt.title("Power vs OD/ID")

plt.figure(2)
plt.plot(desvar, result3[:, 0])
plt.plot(desvar, result3[:, 1])
plt.plot(desvar, result3[:, 2])
plt.plot(desvar, result3[:, 3])
plt.xlabel("Ratio of Motor ID to OD")
plt.ylabel('Power Density (kW/kg)')
plt.title("Power Density vs OD/ID")

plt.figure(3)
plt.plot(desvar, result2[:, 0])
plt.plot(desvar, result2[:, 1])
plt.plot(desvar, result2[:, 2])
plt.plot(desvar, result2[:, 3])
plt.xlabel("Ratio of Motor ID to OD")
plt.ylabel('Resistive Loss (%)')
plt.title("Resistive Loss vs OD/ID")
plt.show()

print('done')
