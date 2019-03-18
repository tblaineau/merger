import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as malptch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))

# def microlens(t, mag, u0, t0, tE):
# 	return mag - 2.5*np.log10(microlensing_amplification(t, u0, t0, tE))

PERIOD_EARTH = 365.2422
alphaS = 80.8941667*np.pi/180.
deltaS = -69.7561111*np.pi/180.
epsilon = (90. - 66.56070833)*np.pi/180.
t_origin = 51442 #(21 septembre 1999) #58747 #(21 septembre 2019)

def parallax(t, mag, u0, t0, tE, delta_u, theta):
	sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
	beta = np.arcsin(sin_beta) #ok because beta is in -pi/2; pi/2
	if abs(beta)==np.pi/2:
		lambda_star = 0
	else:
		lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))
	tau = (t-t0)/tE
	phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
	u_D = np.array([ 
		-u0*np.sin(theta) + tau*np.cos(theta),
		 u0*np.cos(theta) + tau*np.sin(theta)
		])
	u_t = np.array([
		-delta_u*np.sin(phi),
		 delta_u*np.cos(phi)*sin_beta
		])
	# t1 = u0*u0
	# t2 = ((t-t0)/tE)**2
	# t3 = delta_u**2 * (np.cos(phi)**2 + (np.sin(phi) * np.cos(beta))**2)
	# t4 = -delta_u * (t-t0)/tE * (np.cos(theta) * np.cos(phi) + np.cos(beta) * np.sin(theta) * np.sin(phi))
	# t5 = u0 * delta_u * (np.sin(theta) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(beta))
	# u = np.sqrt(t1+t2+t3+t4+t5)
	u = np.linalg.norm(u_D-u_t, axis=0)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def microlens(t, params):
	mag, blend, u0, t0, tE, delta_u, theta,_ = params
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * parallax(t, mag, u0, t0, tE, delta_u, theta))

def projected_plan(t, params):
	mag, blend, u0, t0, tE, delta_u, theta,_ = params
	xD = (t-t0)/tE * np.cos(theta) - u0*np.sin(theta)
	yD = (t-t0)/tE * np.sin(theta) + u0*np.cos(theta)

	sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
	beta = np.arcsin(sin_beta)
	if abs(beta)==np.pi/2:
		lambda_star = 0
	else:
		lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))

	phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star

	x = - delta_u * np.sin(phi)
	y = delta_u * np.cos(phi) * sin_beta
	return xD, yD, x, y

def u_earth(ti, delta_u):
	sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
	beta = np.arcsin(sin_beta)
	if abs(beta)==np.pi/2:
		lambda_star = 0
	else:
		lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))
	phi = 2*np.pi * (ti-t_origin)/PERIOD_EARTH - lambda_star
	print("Lambda :"+ str(lambda_star*180/np.pi))
	print("Beta :" +str(beta*180/np.pi))
	print("Phi : " +str((phi*180/np.pi)%360))
	print("Temps : "+str(ti-t_origin))
	print()
	return [[0, -delta_u*np.sin(phi)], [0, delta_u*np.cos(phi)*sin_beta]]

def u_deflector(ti, theta, tE, u0, t0):
	return [[0, -u0*np.sin(theta)+(ti-t0)/tE*np.cos(theta)], [0, u0*np.cos(theta)+(ti-t0)/tE*np.sin(theta)]]

fig, axs = plt.subplots(1,2)
ax0 = axs[0]
ax1 = axs[1]

fig3 =  plt.figure()
ax3 = fig3.gca()

time = np.arange(48928, 52697, 10)
u0=0.5
t0=50000
tE=500
mag=19
params = {
	'mag':mag,
	'blend':0.,
	'u0':u0,
	't0':t0,
	'tE':tE,
	'delta_u':0,	#no parallax
	'theta':10*np.pi/180.,
	'ti':t0
}

vec_earth, = ax1.plot(*u_earth(t0, params["delta_u"]), marker='o')
deflector_position = u_deflector(t0, params["theta"], params["tE"], params["u0"], params["t0"])
vec_defle, = ax1.plot(*deflector_position, marker='o')
ax3.set_xlim(-2,2)
ax3.set_ylim(-2,2)
ax3.set_aspect('equal')

a = microlens(time, params.values())
line, = ax0.plot(time,  a, color='black', linewidth=0.5)
xD, yD, xpE, ypE = projected_plan(time, params.values())
defl_line = ax1.scatter(xD, yD, c=a, s=10)
earth_projected_orbit = ax1.scatter(xpE, ypE, c=a, s=1)
current_ampli, = ax0.plot(params["ti"], microlens(params["ti"], params.values()), marker="o", markersize=10, color='red')
einstein_radius = malptch.Circle((deflector_position[0][1], deflector_position[1][1]), 1 , fill=False, color="black")


# co_lines = LineCollection(np.reshape(np.column_stack(np.array([xD, yD, xpE, ypE])), (377,2,2)), linewidth=14./a, cmap=plt.get_cmap('Reds_r'), alpha=0.5)
# co_lines.set_array(a)
# ax1.add_collection(co_lines)
ax1.add_patch(einstein_radius)
ax1.axis("equal")

fig2 = plt.figure()

u0slider_ax = fig2.add_axes([0.25, 0.1, 0.65, 0.03])
u0slider = Slider(u0slider_ax, 'u0', 0, 2, valinit=u0)

t0slider_ax = fig2.add_axes([0.25, 0.05, 0.65, 0.03])
t0slider = Slider(t0slider_ax, 't0', 48000, 53000, valinit=t0)

tEslider_ax = fig2.add_axes([0.25, 0.15, 0.65, 0.03])
tEslider = Slider(tEslider_ax, 'tE', -10000, 10000, valinit=tE)

magslider_ax = fig2.add_axes([0.25, 0.2, 0.65, 0.03])
magslider = Slider(magslider_ax, 'mag', 10, 23, valinit=mag)

delta_uslider_ax = fig2.add_axes([0.25, 0.25, 0.65, 0.03])
delta_uslider = Slider(delta_uslider_ax, 'delta_u', 0, 2, valinit=0)

ti_slider_ax = fig2.add_axes([0.25, 0.3, 0.65, 0.03])
ti_slider = Slider(ti_slider_ax, 't_i', 48000, 53000, valinit=params["t0"])

theta_slider_ax = fig2.add_axes([0.25, 0.4, 0.65, 0.03])
theta_slider = Slider(theta_slider_ax, 'theta', 0., 360., valinit=10)

blend_slider_ax = fig2.add_axes([0.25, 0.45, 0.65, 0.03])
blend_slider = Slider(blend_slider_ax, 'blend', 0., 1, valinit=0.)


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

def update_theta(val):
	params["theta"] = val/180.*np.pi
	update_graph()

def update_blend(val):
	params["blend"] = val
	update_graph()

def update_ti(val):
	params['ti'] = val
	update_graph()

def update_graph():
	ydata = microlens(time, params.values())
	line.set_ydata(ydata)
	#ax0.set_ylim(ydata.max()+1, ydata.min()-1)
	
	colnorm = Normalize(vmin=16, vmax=19)
	colmap = ScalarMappable(norm=colnorm, cmap=plt.get_cmap('Reds_r'))

	xD, yD, xpE, ypE = projected_plan(time, params.values())
	earth_projected_orbit.set_offsets(np.column_stack([xpE,ypE]))
	earth_projected_orbit.set_facecolor(colmap.to_rgba(ydata))
	# earth_projected_orbit.set_ydata(ypE)
	defl_line.set_offsets(np.column_stack([xD,yD]))
	defl_line.set_facecolor(colmap.to_rgba(ydata))

	vec_earth.set_data(*u_earth(params["ti"], params["delta_u"]))
	deflector_position = u_deflector(params["ti"], params["theta"], params["tE"], params["u0"], params["t0"])
	vec_defle.set_data(*deflector_position)

	current_ampli.set_data(params["ti"], microlens(params["ti"], params.values()))

	einstein_radius.center = deflector_position[0][1], deflector_position[1][1]

	# co_lines.set_segments(np.reshape(np.column_stack(np.array([xD, yD, xpE, ypE])), (377,2,2)))
	# co_lines.set_array(ydata)

	# defl_line.set_ydata(yD)
	fig.canvas.draw_idle()
	fig3.canvas.draw_idle()

u0slider.on_changed(update_u0)
t0slider.on_changed(update_t0)
tEslider.on_changed(update_tE)
ti_slider.on_changed(update_ti)
magslider.on_changed(update_mag)
delta_uslider.on_changed(update_delta_u)
theta_slider.on_changed(update_theta)
blend_slider.on_changed(update_blend)
ax0.set_ylim(10, 20)
ax0.invert_yaxis()
plt.show()
