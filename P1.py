# BLOQUE I - 1: ECUACIONES DIFERENCIALES ORDINARIAS
# DANIEL CÁRDENAS HERNÁNDEZ 74396404-Y
# MÉTODO DE VERLET Y EXTRAS

import numpy as np
import matplotlib.pyplot as plt

# Definición de funciones
def f(x,y):
    r = np.hypot(x,y)
    return -G*M*x/r**3,-G*M*y/r**3
def mod(x,y):
    return np.hypot(x,y)

# Definición de ctes
G = 6.6738e-11                  # Cte gravitación universal (Nm^2/kg)
M = 1.9891e30                   # Masa del Sol (kg)
m = 5.9722e24                   # Masa de la Tierra (kg)
t0 = 0                          # Tiempo inicial (s)
t1 = 157680000                  # Tiempo final (s)
h = 3600                        # Paso temporal (s)
N = int((t1/h)+1)               
t = np.linspace(t0,t1,N)

# MÉTODO DE EULER---------------------------------------------------------------
# Inicialización de vectores
xe = np.empty(N);ye = np.empty(N)
vxe = np.empty(N);vye = np.empty(N)

# Condiciones iniciales
xe[0] = 1.4719e11;ye[0] = 0      # Posición inicial (m)                 
vxe[0] = 0;vye[0] = 3.0287e4     # Velocidad inicial tg (m/s)

# Iteraciones
for i in range(0,N-1):
    xe[i+1] = xe[i]+h*vxe[i]
    ye[i+1] = ye[i]+h*vye[i]
    vxe[i+1] = vxe[i]+h*f(xe[i],ye[i])[0]
    vye[i+1] = vye[i]+h*f(xe[i],ye[i])[1]  

# Energías 
Ec = 0.5*m*(mod(vxe,vye))**2     # Energía cinética
Ep = -(G*M*m)/mod(xe,ye)         # Energía potencial
Et = Ep+Ec                       # Energía total

# Representaciones
plt.figure(figsize=[9,7])
plt.subplot(2,1,1)
plt.xlabel('t(s)')
plt.ylabel('R(m)')
plt.title('Método Euler')
plt.plot(t,mod(xe,ye),'purple')
plt.subplot(2,2,3)
plt.xlabel('t(s)')
plt.ylabel('E(J)')
plt.plot(t,Ec,label='Ec')
plt.plot(t,Ep,label='Ep')
plt.plot(t,Et,label='Et')
plt.legend(loc='best')
plt.subplot(2,2,4)
plt.xlabel('y(m)')
plt.ylabel('x(m)')
plt.plot(ye,xe,'r')
plt.axis('square')
plt.show()

# MÉTODO DE RUNGE-KUTTA DE ORDEN 2----------------------------------------------
# Inicialización de vectores
x2 = np.empty(N);y2 = np.empty(N)
vx2 = np.empty(N);vy2 = np.empty(N)

# Condiciones iniciales
x2[0] = 1.4719e11;y2[0] = 0                     
vx2[0] = 0;vy2[0] = 3.0287e4

# Iteraciones
for i in range(0,N-1):
    k1x = h*vx2[i];k1y = h*vy2[i]
    l1x = h*f(x2[i],y2[i])[0];l1y = h*f(x2[i],y2[i])[1]
    k2x = k1x+h*l1x/2;k2y = k1y+h*l1y/2
    l2x = h*f(x2[i]+1/2*k1x,y2[i]+1/2*k1y)[0]
    l2y = h*f(x2[i]+1/2*k1x,y2[i]+1/2*k1y)[1]
    x2[i+1] = x2[i]+k2x;y2[i+1] = y2[i]+k2y
    vx2[i+1] = vx2[i]+l2x;vy2[i+1] = vy2[i]+l2y

# Energías
Ec = 0.5*m*(mod(vx2,vy2))**2      
Ep = -(G*M*m)/mod(x2,y2)          
Et = Ep+Ec                       

# Representaciones
plt.figure(figsize=[9,7])
plt.subplot(2,1,1)
plt.xlabel('t(s)')
plt.ylabel('R(m)')
plt.title('Método Runge-Kutta de orden 2')
plt.plot(t,mod(x2,y2),'purple')
plt.subplot(2,2,3)
plt.xlabel('t(s)')
plt.ylabel('E(J)')
plt.plot(t,Ec,label='Ec')
plt.plot(t,Ep,label='Ep')
plt.plot(t,Et,label='Et')
plt.legend(loc='best')
plt.subplot(2,2,4)
plt.xlabel('y(m)')
plt.ylabel('x(m)')
plt.plot(y2,x2,'r')
plt.axis('square')
plt.show()

#MÉTODO DE RUNGE-KUTTA DE ORDEN 4-----------------------------------------------
# Inicialización de vectores
x4 = np.empty(N);y4 = np.empty(N)
vx4 = np.empty(N);vy4 = np.empty(N)

# Condiciones iniciales
x4[0] = 1.4719e11;y4[0] = 0
vx4[0] = 0;vy4[0] = 3.0287e4

# Iteraciones
for i in range(0,N-1):
    k1x4 = h*vx4[i];k1y4 = h*vy4[i]
    l1x4 = h*f(x4[i],y4[i])[0];l1y4 = h*f(x4[i],y4[i])[1]
    k2x4 = k1x4+h*l1x4/2;k2y4 = k1y4+h*l1y4/2
    l2x4 = h*f(x4[i]+1/2*k1x4,y4[i]+1/2*k1y4)[0]
    l2y4 = h*f(x4[i]+1/2*k1x4,y4[i]+1/2*k1y4)[1]
    k3x4 = k1x4+h*l2x4/2;k3y4 = k1y4+h*l2y4/2
    l3x4 = h*f(x4[i]+1/2*k2x4,y4[i]+1/2*k2y4)[0]
    l3y4 = h*f(x4[i]+1/2*k2x4,y4[i]+1/2*k2y4)[1]
    k4x4 = k1x4+h*l3x4;k4y4 = k1y4+h*l3y4
    l4x4 = h*f(x4[i]+k3x4,y4[i]+k3y4)[0]
    l4y4 = h*f(x4[i]+k3x4,y4[i]+k3y4)[1]
    x4[i+1] = x4[i]+1/6*(k1x4+2*k2x4+2*k3x4+k4x4)
    y4[i+1] = y4[i]+1/6*(k1y4+2*k2y4+2*k3y4+k4y4)
    vx4[i+1] = vx4[i]+1/6*(l1x4+2*l2x4+2*l3x4+l4x4)
    vy4[i+1] = vy4[i]+1/6*(l1y4+2*l2y4+2*l3y4+l4y4)

# Energías
Ec = 0.5*m*(mod(vx4,vy4))**2  
Ep = -(G*M*m)/mod(x4,y4)                 
Et = Ep+Ec                       

#Representaciones
plt.figure(figsize=[9,7])
plt.subplot(2,1,1)
plt.xlabel('t(s)')
plt.ylabel('R(m)')
plt.title('Método Runge-Kutta de orden 4')
plt.plot(t,mod(x4,y4),'purple')
plt.subplot(2,2,3)
plt.xlabel('t(s)')
plt.ylabel('E(J)')
plt.plot(t,Ec,label='Ec')
plt.plot(t,Ep,label='Ep')
plt.plot(t,Et,label='Et')
plt.legend(loc='best')
plt.subplot(2,2,4)
plt.xlabel('y(m)')
plt.ylabel('x(m)')
plt.plot(y4,x4,'r')
plt.axis('square')
plt.show()

# MÉTODO DE VERLET DE VELOCIDADES-----------------------------------------------
xd = np.empty(N);yd = np.empty(N)
vxd = np.empty(N);vyd = np.empty(N)
axd = np.empty(N);ayd = np.empty(N)

# Condiciones iniciales
xd[0] = 1.4719e11;yd[0] = 0                     
vxd[0] = 0;vyd[0] = 3.0287e4
axd[0], ayd[0] = f(xd[0], yd[0])

# Iteraciones
for i in range(0,N-1):
    xd[i+1] = xd[i]+h*vxd[i]+(h**2/2)*f(xd[i],yd[i])[0]
    yd[i+1] = yd[i]+h*vyd[i]+(h**2/2)*f(xd[i],yd[i])[1]
    vxd[i+1] = vxd[i]+(h/2)*(f(xd[i],yd[i])[0]+f(xd[i+1],yd[i+1])[0])
    vyd[i+1] = vyd[i]+(h/2)*(f(xd[i],yd[i])[1]+f(xd[i+1],yd[i+1])[1])

# Energías
Ec = 0.5*m*(mod(vxd,vyd))**2  
Ep = -(G*M*m)/mod(xd,yd)               
Et = Ep+Ec

# Representaciones
plt.figure(figsize=[9,7])
plt.subplot(2,1,1)
plt.xlabel('t(s)')
plt.ylabel('R(m)')
plt.title('Método Verlet de velocidades')
plt.plot(t,mod(xd,yd),'purple')
plt.subplot(2,2,3)
plt.xlabel('t(s)')
plt.ylabel('E(J)')
plt.plot(t,Ec,label='Ec')
plt.plot(t,Ep,label='Ep')
plt.plot(t,Et,label='Et')
plt.legend(loc='best')
plt.subplot(2,2,4)
plt.xlabel('y(m)')
plt.ylabel('x(m)')
plt.plot(yd,xd,'r')
plt.axis('square')
plt.show()

# MÉTODO DE VERLET--------------------------------------------------------------
# Inicialización de vectores
xv = np.empty(N);yv = np.empty(N)
vxv = np.empty(N);vyv = np.empty(N)
vx1v = np.empty(N);vy1v = np.empty(N)

# Condiciones iniciales
xv[0] = 1.4719e11;yv[0] = 0                     
vx0v = 0;vy0v = 3.0287e4
vxv[0] = vx0v+1/2*h*f(xv[0],yv[0])[0]
vyv[0] = vy0v+1/2*h*f(xv[0],yv[0])[1]
vx1v[0] = 0;vy1v[0] = 3.0287e4

# Iteraciones
for i in range(0,N-1):            
    xv[i+1] = xv[i]+h*vxv[i];yv[i+1] = yv[i]+h*vyv[i]
    kxv = h*f(xv[i+1],yv[i+1])[0];vxv[i+1] = vxv[i]+kxv
    kyv = h*f(xv[i+1],yv[i+1])[1];vyv[i+1] = vyv[i]+kyv
    vx1v[i+1] = vxv[i]+kxv/2;vy1v[i+1] = vyv[i]+kyv/2

# Energías
Ec = 0.5*m*(mod(vx1v,vy1v))**2 
Ep = -(G*M*m)/mod(xv,yv)                 
Et = Ep+Ec                       

# Representaciones
plt.figure(figsize=[9,7])
plt.subplot(2,1,1)
plt.xlabel('t(s)')
plt.ylabel('R(m)')
plt.title('Método Verlet')
plt.plot(t,mod(xv,yv),'purple')
plt.subplot(2,2,3)
plt.xlabel('t(s)')
plt.ylabel('E(J)')
plt.plot(t,Ec,label='Ec')
plt.plot(t,Ep,label='Ep')
plt.plot(t,Et,label='Et')
plt.legend(loc='best')
plt.subplot(2,2,4)
plt.xlabel('y(m)')
plt.ylabel('x(m)')
plt.plot(yv,xv,'r')
plt.axis('square')
plt.show()
