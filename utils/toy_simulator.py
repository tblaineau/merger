import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as malptch

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))

# def microlens(t, mag, u0, t0, tE):
# 	return mag - 2.5*np.log10(microlensing_amplification(t, u0, t0, tE))

def parallax(t, mag, u0, t0, tE, delta_u, beta, psi, alpha0):
	phi = 2*np.pi/365.25 * (t-t0) + alpha0
	t1 = u0*u0
	t2 = ((t-t0)/tE)**2
	t3 = delta_u**2 * (np.cos(phi)**2 + (np.sin(phi) * np.cos(beta))**2)
	t4 = -delta_u * (t-t0)/tE * (np.cos(psi) * np.cos(phi) + np.cos(beta) * np.sin(psi) * np.sin(phi))
	t5 = u0 * delta_u * (np.sin(psi) * np.cos(phi) - np.cos(psi) * np.sin(phi) * np.cos(beta))
	u = np.sqrt(t1+t2+t3+t4+t5)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def microlens(t, mag, u0, t0, tE, delta_u, beta, psi, alpha0):
	return mag - 2.5*np.log10(parallax(t, mag, u0, t0, tE, delta_u, beta, psi, alpha0))

def projected_plan(t, mag, u0, t0, tE, delta_u, beta, psi, alpha0):
	xD = (t-t0)/tE * np.cos(psi) - u0*np.sin(psi)
	yD = (t-t0)/tE * np.sin(psi) + u0*np.cos(psi)

	phi = 2*np.pi/365.25 * (t-t0) + alpha0
	x = delta_u * np.cos(phi)
	y = delta_u * np.sin(phi) * np.cos(beta)
	return xD, yD, x, y

fig, axs = plt.subplots(1,2)
ax0 = axs[0]
ax1 = axs[1]


time = np.arange(48928, 52697, 10)
u0=0.5
t0=50000
tE=500
mag=19
params = {
	'mag':mag,
	'u0':u0,
	't0':t0,
	'tE':tE,
	'delta_u':0,	#no parallax
	'beta':10*np.pi/180.,
	'psi':10*np.pi/180.,
	'alpha0':2.*np.pi/3.
}

line, = ax0.plot(time,  microlens(time, *params.values()), color='black', linewidth=0.5)
xD, yD, xpE, ypE = projected_plan(time, *params.values())
defl_line, = ax1.plot(xD, yD)
earth_projected_orbit, = ax1.plot(xpE, ypE)
ax1.add_patch(malptch.Circle((0,0),1, fill=False, color="black"))
ax1.axis("equal")

fig2 = plt.figure()

u0slider_ax = fig2.add_axes([0.25, 0.1, 0.65, 0.03])
u0slider = Slider(u0slider_ax, 'u0', 0, 2, valinit=u0)

t0slider_ax = fig2.add_axes([0.25, 0.05, 0.65, 0.03])
t0slider = Slider(t0slider_ax, 't0', 48000, 53000, valinit=t0)

tEslider_ax = fig2.add_axes([0.25, 0.15, 0.65, 0.03])
tEslider = Slider(tEslider_ax, 'tE', 100, 10000, valinit=tE)

magslider_ax = fig2.add_axes([0.25, 0.2, 0.65, 0.03])
magslider = Slider(magslider_ax, 'mag', 10, 23, valinit=mag)

delta_uslider_ax = fig2.add_axes([0.25, 0.25, 0.65, 0.03])
delta_uslider = Slider(delta_uslider_ax, 'delta_u', 0, 2, valinit=0)

beta_slider_ax = fig2.add_axes([0.25, 0.3, 0.65, 0.03])
beta_slider = Slider(beta_slider_ax, 'beta', 0., np.pi, valinit=10*np.pi/180.)

alpha0_slider_ax = fig2.add_axes([0.25, 0.35, 0.65, 0.03])
alpha0_slider = Slider(alpha0_slider_ax, 'alpha0', 0., 2*np.pi, valinit=10*np.pi/180.)

psi_slider_ax = fig2.add_axes([0.25, 0.4, 0.65, 0.03])
psi_slider = Slider(psi_slider_ax, 'psi', 0., 2*np.pi, valinit=10*np.pi/180.)

def update_u0(val):
	params["u0"] = val
	update_graph()

def update_t0(val):
	params["t0"] = val
	update_graph()

def update_tE(val):
	params["tE"] = val
	update_graph()

def update_mag(val):
	params["mag"] = val
	update_graph()

def update_delta_u(val):
	params["delta_u"] = val
	update_graph()

def update_beta(val):
	params["beta"] = val
	update_graph()

def update_alpha0(val):
	params["alpha0"] = val
	update_graph()

def update_psi(val):
	params["psi"] = val
	update_graph()

def update_graph():
	ydata = microlens(time, *params.values())
	line.set_ydata(ydata)
	ax0.set_ylim(ydata.max()+1, ydata.min()-1)

	xD, yD, xpE, ypE = projected_plan(time, *params.values())
	earth_projected_orbit.set_xdata(xpE)
	earth_projected_orbit.set_ydata(ypE)
	defl_line.set_xdata(xD)
	defl_line.set_ydata(yD)
	fig.canvas.draw_idle()
	
u0slider.on_changed(update_u0)
t0slider.on_changed(update_t0)
tEslider.on_changed(update_tE)
magslider.on_changed(update_mag)
delta_uslider.on_changed(update_delta_u)
beta_slider.on_changed(update_beta)
alpha0_slider.on_changed(update_alpha0)
psi_slider.on_changed(update_psi)
ax0.set_ylim(16, 20)
ax0.invert_yaxis()
plt.show()