import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def microlens(t, mag, u0, t0, tE):
	return mag - 2.5*np.log10(microlensing_amplification(t, u0, t0, tE))

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.25)

time = np.arange(48928, 52697, 10)
u0=0.5
t0=50000
tE=500
mag=19
params = [mag, u0, t0, tE]

line, = ax.plot(time,  microlens(time, mag, u0, t0, tE), color='black', linewidth=0.5)

u0slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
u0slider = Slider(u0slider_ax, 'u0', 0, 2, valinit=u0)

t0slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
t0slider = Slider(t0slider_ax, 't0', 48000, 53000, valinit=t0)

tEslider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
tEslider = Slider(tEslider_ax, 'tE', 100, 10000, valinit=tE)

magslider_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
magslider = Slider(magslider_ax, 'mag', 10, 23, valinit=mag)


def update_u0(val):
	params[1] = val
	update_graph()

def update_t0(val):
	params[2] = val
	update_graph()

def update_tE(val):
	params[3] = val
	update_graph()

def update_mag(val):
	params[0] = val
	update_graph()

def update_graph():
	ydata = microlens(time, *params)
	line.set_ydata(ydata)
	ax.set_ylim(ydata.max()+1, ydata.min()-1)
	fig.canvas.draw_idle()
	
u0slider.on_changed(update_u0)
t0slider.on_changed(update_t0)
tEslider.on_changed(update_tE)
magslider.on_changed(update_mag)
ax.set_ylim(16, 20)
ax.invert_yaxis()
plt.show()