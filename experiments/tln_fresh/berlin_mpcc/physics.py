"""Check candidate + backup braking with the unchanged project PID/RK4 dynamics.
Uses static map, observed LiDAR foreground and remembered perceived cubes only.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,yaml,pyglet
pyglet.options['headless']=True
from numba import njit
from f110_gym.envs.dynamic_models import pid,vehicle_dynamics_st
@njit(cache=True)
def step(state,delta,speed,p):
 accel,sv=pid(speed,delta,state[3],state[2],p[11],p[13],p[15],p[14]);u=np.array([sv,accel]);k1=vehicle_dynamics_st(state,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);k2=vehicle_dynamics_st(state+.005*k1,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);k3=vehicle_dynamics_st(state+.005*k2,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);k4=vehicle_dynamics_st(state+.01*k3,u,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);result=state+.01*(1./6.)*(k1+2*k2+2*k3+k4)
 if result[4]>2*np.pi:result[4]-=2*np.pi
 elif result[4]<0:result[4]+=2*np.pi
 return result
