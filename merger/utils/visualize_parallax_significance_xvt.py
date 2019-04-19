import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

from scipy.stats import rv_continuous
from scipy.integrate import dblquad
from scipy.misc import derivative

from mpl_toolkits.mplot3d import Axes3D

from astropy import constants
from astropy import units

COLOR_FILTERS = {
    "red_E": {"mag": "red_E", "err": "rederr_E"},
    "red_M": {"mag": "red_M", "err": "rederr_M"},
    "blue_E": {"mag": "blue_E", "err": "blueerr_E"},
    "blue_M": {"mag": "blue_M", "err": "blueerr_M"},
}

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

PERIOD_EARTH = 365.2422
alphaS = 80.8941667 * np.pi / 180.0
deltaS = -69.7561111 * np.pi / 180.0
epsilon = (90.0 - 66.56070833) * np.pi / 180.0
t_origin = 51442  # (21 septembre 1999) #58747 #(21 septembre 2019)


def parallax(t, mag, u0, t0, tE, delta_u, theta):
    sin_beta = np.cos(epsilon) * np.sin(deltaS) - np.sin(epsilon) * np.cos(
        deltaS
    ) * np.sin(alphaS)
    beta = np.arcsin(sin_beta)  # ok because beta is in -pi/2; pi/2
    if abs(beta) == np.pi / 2:
        lambda_star = 0
    else:
        lambda_star = np.sign(
            (
                np.sin(epsilon) * np.sin(deltaS)
                + np.cos(epsilon) * np.sin(alphaS) * np.cos(deltaS)
            )
            / np.cos(beta)
        ) * np.arccos(np.cos(deltaS) * np.cos(alphaS) / np.cos(beta))
    tau = (t - t0) / tE
    phi = 2 * np.pi * (t - t_origin) / PERIOD_EARTH - lambda_star
    u_D = np.array(
        [
            -u0 * np.sin(theta) + tau * np.cos(theta),
            u0 * np.cos(theta) + tau * np.sin(theta),
        ]
    )
    u_t = np.array([-delta_u * np.sin(phi), delta_u * np.cos(phi) * sin_beta])
    # t1 = u0*u0
    # t2 = ((t-t0)/tE)**2
    # t3 = delta_u**2 * (np.cos(phi)**2 + (np.sin(phi) * np.cos(beta))**2)
    # t4 = -delta_u * (t-t0)/tE * (np.cos(theta) * np.cos(phi) + np.cos(beta) * np.sin(theta) * np.sin(phi))
    # t5 = u0 * delta_u * (np.sin(theta) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(beta))
    # u = np.sqrt(t1+t2+t3+t4+t5)
    u = np.linalg.norm(u_D - u_t, axis=0)
    return (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))


def microlens(t, params):
    mag, blend, u0, t0, tE, delta_u, theta = params
    return -2.5 * np.log10(
        blend * np.power(10, mag / -2.5)
        + (1 - blend)
        * np.power(10, mag / -2.5)
        * parallax(t, mag, u0, t0, tE, delta_u, theta)
    )


def microlens_simple(t, params):
    mag, blend, u0, t0, tE, delta_u, theta = params
    u = np.sqrt(u0 * u0 + ((t - t0) ** 2) / tE / tE)
    A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    return -2.5 * np.log10(
        blend * np.power(10, mag / -2.5) + (1 - blend) * np.power(10, mag / -2.5) * A
    )


def rho_halo(x):
    """pdf of dark halo density
	
	[description]
	
	Arguments:
		x {float} -- x = d_OD/d_OS
	
	Returns:
		{float} -- dark matter density at x
	"""
    a = 5000.0  # pc
    rho_0 = 0.0079  # M_sol/pc^3
    d_sol = 8500  # pc
    l_lmc, b_lmc = 280.4652 / 180.0 * np.pi, -32.8884 / 180.0 * np.pi
    r_lmc = 55000  # pc
    cosb_lmc = np.cos(b_lmc)
    cosl_lmc = np.cos(l_lmc)

    A = d_sol ** 2 + a ** 2
    B = d_sol * cosb_lmc * cosl_lmc

    return rho_0 * A / ((x * r_lmc) ** 2 - 2 * x * r_lmc * B + A)


def p_x(x):
    return rho_halo(x) * np.sqrt(x * (1 - x))


def p_vt(v_T, v0=220):
    # Proba to find v_T i
    return (2 * v_T / (v0 ** 2)) * np.exp(-v_T ** 2 / (v0 ** 2))


r_lmc = 55000
MASS = 100
r_0 = (
    np.sqrt(4 * constants.G / (constants.c ** 2) * r_lmc * units.pc)
    .decompose([units.Msun, units.pc])
    .value
)
kms_to_pcd = (units.km / units.s).to(units.pc / units.day)
r_earth = (150 * 1e6) * units.km.to(units.pc)


def tE_from_xvt(x, v_T):
    R_E = r_0 * np.sqrt(MASS * x * (1 - x))
    tE = R_E / (v_T * kms_to_pcd)
    return tE


def delta_u_from_x(x, mass=MASS):
    r = r_earth / (r_0 * np.sqrt(mass))
    return r * np.sqrt((1 - x) / x)


time_range = np.linspace(48928, 52697, 10000)
t0 = 50000
tE = 500
mag = 19
BASE_U0 = 0.8
MAX_DELTA_U = 0.02

params = {
    "mag": mag,
    "blend": 0.0,
    "u0": BASE_U0,
    "t0": t0,
    "tE": 300,
    "delta_u": 0.5,  # no parallax
    "theta": 10 * np.pi / 180.0,
}
x = np.linspace(0.01, 0.99, 100)
vt = np.linspace(-601, 600, 100)

delta_u = delta_u_from_x(x)
tE = tE_from_xvt(x, vt)
params_set = [
    params["mag"],
    params["blend"],
    params["u0"],
    params["t0"],
    tE[None, :, None],
    delta_u[None, None, :],
    params["theta"],
]
st1 = time.time()
absolute_diffs = np.abs(
    microlens(time_range[:, None, None], params_set)
    - microlens_simple(time_range[:, None, None], params_set)
)
print(absolute_diffs.shape)
max_diff = absolute_diffs.mean(axis=0)
print(max_diff.shape)
print(time.time() - st1)


def on_click(event):
    print(event.x, event.y, event.xdata, event.ydata)
    params = {
        "mag": mag,
        "blend": 0.0,
        "u0": BASE_U0,
        "t0": t0,
        "tE": event.ydata,
        "delta_u": event.xdata,
        "theta": 90 * np.pi / 180.0,
    }
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(time_range, microlens(time_range, params.values()))
    axs[0].plot(time_range, microlens_simple(time_range, params.values()))
    axs[0].invert_yaxis()
    axs[1].plot(
        time_range,
        microlens(time_range, params.values())
        - microlens_simple(time_range, params.values()),
    )
    axs[1].invert_yaxis()
    fig.suptitle(r"$t_E = $" + str(event.ydata) + r", $\delta_u = $" + str(event.xdata))
    plt.show()


fig = plt.figure()

plt.imshow(
    max_diff,
    origin="lower",
    interpolation="nearest",
    cmap="plasma",
    extent=[x[0], x[-1], vt[0], vt[-1]],
    aspect="auto",
)
col = plt.colorbar()

prob_field2 = p_x(x)[None, :] * p_vt(vt)[:, None]
plt.contour(x, vt, prob_field2, levels=10)
# col2 = plt.colorbar()
# print(prob_field2.max(), prob_field2.min())

plt.xlabel(r"$x$")
plt.ylabel(r"$v_T$")
col.set_label("mean magnitude difference")
fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()
