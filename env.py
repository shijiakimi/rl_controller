import numpy as np
import pyglet
import math 


class ArmEnv(object):
    viewer = None
    dt = .05
    action_bound = [-5, 5]
    goal = {'x': 200., 'y': 200., 'z': 200, 'l': 40}
    state_dim = 5
    action_dim = 2
    inertia_mat = [[0.007, 0., 0.], [0., 0.007, 0.], [0., 0., 0.012]]
    inv_inertia_mat = [[1./0.007, 0., 0.], [0., 1./0.007, 0.], [0., 0., 1./0.012]]
    transThrust2GenForce = np.array([[0., 0., 0.17, -0.17],
        					[-0.17, 0.17, 0., 0.],
        					[0.016, 0.016, -0.016, -0.016],
        					[1., 1., 1., 1.]])
    gravity = np.array([0., 0., -9.81])
    mass = 0.665

    def __init__(self):

        self.uav_ori = np.array([1., 0., 0., 0.])
        self.uav_pos = np.zeros(2)
        self.uav_v = np.zeros(3)
        self.uav_w = np.zeros(3)
        self.on_goal = 0
	
    def step(self, action):
        done = False
	'''
        action = np.square(action) * 8.5486

        action = np.clip(action, *self.action_bound)
	#print action
        gene_force = list(np.dot(self.transThrust2GenForce, action))
        B_torque = list(gene_force[0:3]);
    	B_force = np.array([0.0, 0.0, gene_force[3]]);
    	rot_matr = self.quat_to_R(self.uav_ori)
    	acce_lin = np.dot(rot_matr, B_force) / self.mass + self.gravity
    	w_B = np.dot(np.transpose(rot_matr), self.uav_w)
    	acce_ang = np.dot(rot_matr, np.dot(self.inv_inertia_mat, np.array(B_torque) - np.cross(w_B, np.dot(self.inertia_mat, w_B))))
	'''
	  
	#self.uav_v += acce_lin * self.dt
	


	#action = np.clip(action, *self.action_bound)
	#print action	
	#self.uav_v += action[:3] * self.dt
    	
	#self.uav_w += acce_ang * self.dt
    	#self.uav_ori = self.update_ori(self.uav_ori, self.uav_w, self.dt)
    	#self.uav_ori = self.normalize_ori(self.uav_ori)
    	self.uav_pos += action[:2] * self.dt
    	
    	
    	
    	#R_ * (inertiaInv_ * (B_torque - w_B_.cross(inertia_ * w_B_)))
    	

        dist1 = [(self.goal['x'] - self.uav_pos[0]), (self.goal['y'] - self.uav_pos[1])]
	#print dist1
        #r = -np.sqrt(dist1[0]**2+dist1[1]**2)
	#r = -(dist1[0]**2+dist1[1]**2)
	r = np.tanh(1 - 0.005 * (abs(dist1[0]) + abs(dist1[1])))
	if self.uav_pos[0] > 400 or self.uav_pos[1] > 400 or self.uav_pos[0] < 0 or self.uav_pos[1] < 0:
		r -= 10
	#r = 1./(dist1[0] + 0.001) + 1./(dist1[1] + 0.001) + 1./(dist1[2] + 0.001)
        # done and reward
        if self.goal['x'] - self.goal['l']/2 < self.uav_pos[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < self.uav_pos[1] < self.goal['y'] + self.goal['l']/2:
		#if self.goal['z'] - self.goal['l']/2 < self.uav_pos[2] < self.goal['z'] + self.goal['l']/2:
		    r += 10
                    self.on_goal += 1
                    if self.on_goal > 10:

                        done = True
        else:
            self.on_goal = 0
	#print self.on_goal
        # state
        #s = np.concatenate((self.uav_ori, self.uav_pos, self.uav_v, self.uav_w, dist1, [1. if self.on_goal else 0.]))
	s = np.concatenate((self.uav_pos, dist1, [1. if self.on_goal else 0.]))
	#print 'pos: ', self.uav_pos
        return s, r, done

    def reset(self):
        #self.arm_info = 2 * np.pi * np.random.rand(self.state_dim)
	#self.viewer = None

        self.uav_ori = 0.5 * np.random.rand(4)#np.random.rand(4)
        self.uav_pos = 200 * np.random.rand(2)#200 * np.random.rand(3)
        self.uav_v = np.random.rand(3)
        self.uav_w = np.random.rand(3)
        self.on_goal = 0
        #(a1l, a2l) = self.arm_info['l']  # radius, arm length
        #(a1r, a2r) = self.arm_info['r']  # radian, angle
        #a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        #a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        #finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - self.uav_pos[0])/400, (self.goal['y'] - self.uav_pos[1])/400]
        #dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((self.uav_pos, dist1, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
	    #print self.uav_pos
            self.viewer = Viewer(self.uav_pos, self.goal)
        self.viewer.render(self.uav_pos)


	
    def quat_to_R(self, quat):
	qw = quat[0]
	qx = quat[1]
	qy = quat[2]
	qz = quat[3]
	rot = [[0 for i in range(3)] for j in range(3)]
	rot[0][0] = 1 - 2*qy*qy - 2*qz*qz
	rot[0][1] = 2*qx*qy - 2*qz*qw
	rot[0][2] = 2*qx*qz + 2*qy*qw
	rot[1][0] = 2*qx*qy + 2*qz*qw
	rot[1][1] = 1 - 2*qx*qx -2*qz*qz
	rot[1][2] = 2*qy*qz - 2*qx*qw
	rot[2][0] = 2*qx*qz - 2*qy*qw
	rot[2][1] = 2*qy*qz + 2*qx*qw
	rot[2][2] = 1 - 2*qx*qx - 2*qy*qy
	return np.array(rot)


    def update_ori(self, ori, w, dt):
	qw = ori[0]
	qx = ori[1]
	qy = ori[2]
	qz = ori[3]
	wx = w[0]
	wy = w[1]
	wz = w[2]
	qw += 0.5 * (-qx * wx - qy * wy - qz * wz) * dt
	qx += 0.5 * (qw * wx - qz * wy + qy * wz) * dt
	qy += 0.5 * (qz * wx + qw * wy - qx * wz) * dt
	qz += 0.5 * (-qy * wx + qx * wy + qw * wz) * dt
	return np.array([qw, qx, qy, qz])

	
    def normalize_ori(self, ori):
	qw = ori[0]
	qx = ori[1]
	qy = ori[2]
	qz = ori[3]
	norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
	if math.isnan(norm):
	    print 'normalization of ori: ', qw, qx, qy, qz
	return ori/norm
		


class Viewer(pyglet.window.Window):


    def __init__(self, uav_pos, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.uav_pos = uav_pos
        self.center_coord = np.array([200, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.uav = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [200, 200,                # location
                     200, 220,
                     220, 220,
                     220, 200]),
            ('c3B', (249, 86, 86) * 4,))    # color
        '''
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))
		'''
    def render(self, uav_pos):
        self._update_uav(uav_pos)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_uav(self, uav_pos):

	xy01 = uav_pos[:2] - np.array([10, 10])
	xy02 = uav_pos[:2] - np.array([10, -10])
	xy11 = uav_pos[:2] - np.array([-10, -10])
	xy12 = uav_pos[:2] - np.array([-10, 10])
	#print self.uav_pos
	#print xy01, xy02, xy11, xy12
        self.uav.vertices = np.concatenate((xy01, xy02, xy11, xy12))



if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())
