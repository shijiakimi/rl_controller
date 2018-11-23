import numpy as np
import math


# roll, pitch, yaw = theta = [phi, theta, psi]
# adjacent propellers are oriented opposite each other
class ArmEnv(object):
    dt = .02
    action_bound = [0, 1]
    action_clip = [1, 5000]
    goal = {'x': 0, 'y': 0, 'z': 50, 'l': 2}
    state_dim = 13
    action_dim = 4
    gravity = np.array([0., 0., -9.81])
    mass = 0.985
    rho = 1.2
    C_d = 0.3
    l_to_rotor = 0.4
    propeller_size = 0.1
    width = .51
    length = .51
    height = 0.235
    areas = [length * height, width * height, width * length]
    Ix = 1 / 12. * mass * (height ** 2 + width ** 2)
    Iy = 1 / 12. * mass * (height ** 2 + length ** 2)
    Iz = 1 / 12. * mass * (width ** 2 + length ** 2)
    moments_of_inertia = [Ix, Iy, Iz]

    def __init__(self, init_pos, init_ori, init_v, init_w):

        self.uav_init_euler = np.array(init_ori)
        self.uav_init_v = np.array(init_v)
        self.uav_init_w = np.array(init_w)
        self.uav_euler = np.array(list(init_ori))
        self.uav_pos = np.array(list(init_pos))
        self.uav_init_pos = np.array(init_pos)
        self.uav_v = np.array(list(init_v))
        self.uav_w = np.array(list(init_w))
        self.prop_wind_speed = np.zeros(4)
        self.on_goal = 0
        self.time = 0
        #print self.uav_init_pos, self.uav_init_euler, self.uav_init_v, self.uav_init_w

    def step(self, action):
        done = False
        # action = np.clip(action, self.action_clip[0], self.action_clip[1])
        # print 'action', action

        #action = [action_[0]] * 4
        self.get_prop_wind_speed()
        thrusts = self.get_thrust(action)
        #print 'thrust', thrusts
        linear_acc = self.get_linear_forces(thrusts) / self.mass
        print 'linear_acc', linear_acc
        self.uav_pos += self.uav_v * self.dt + 0.5 * linear_acc * self.dt ** 2
        #print 'pos', self.uav_pos
        self.uav_v += linear_acc * self.dt
        moments = self.get_moments(thrusts)

        angular_acc = moments / self.moments_of_inertia
        self.uav_euler += self.uav_w * self.dt + 0.5 * angular_acc * self.dt ** 2
        self.uav_euler = (self.uav_euler + 2 * np.pi) % (2 * np.pi)
        self.uav_w += angular_acc * self.dt


        # new_uav_v = self.uav_v + action
        # self.uav_pos = (new_uav_v ** 2 - self.uav_v ** 2) / (2 * self.dt)
        # self.uav_v = new_uav_v
        #self.uav_pos += [0,0, action[0]*self.dt]
        #self.time += self.dt
        dist1 = [(self.goal['x'] - self.uav_pos[0]), (self.goal['y'] - self.uav_pos[1]),
                 (self.goal['z'] - self.uav_pos[2])]
        r = -np.linalg.norm(dist1)
        # r = -math.sqrt(dist1[0] ** 2) - math.sqrt(dist1[1] ** 2) - math.sqrt(dist1[2] ** 2)
        # r = - abs(dist1[2]) - abs(dist1[1]) - abs(dist1[0])
        #r = -min(abs(self.goal['z'] - self.uav_pos[2]), 20.0)
        # r /= 3
        # r = np.tanh(1-1.0/50 * (abs(dist1[0]) + abs(dist1[1]) + abs(dist1[2])))
        # rp = 0.004 * math.sqrt(dist1[0]**2 + dist1[1]**2 + dist1[2]**2)
        # rv = 0.0005 * math.sqrt(self.uav_v[0]**2 + self.uav_v[1]**2 + self.uav_v[2]**2)
        # rw = 0.0003 * math.sqrt(self.uav_w[0] ** 2 + self.uav_w[1] ** 2 + self.uav_w[2] ** 2)
        # r = -rp - rv - rw
        # r = -abs(self.uav_pos[2] - self.goal['z'])
        # r = np.tanh(1-1.0/50 * (abs(dist1[0]) + abs(dist1[1])))
        # r = np.tanh(1 - 1.0/(self.goal['x'] + self.goal['y'] + self.goal['z']) * (abs(dist1[0]) + abs(dist1[1]) + abs(dist1[2])))
        # if self.uav_pos[0] > 2*self.goal['x'] or self.uav_pos[1] > 2*self.goal['y'] or self.uav_pos[2] > 2*self.goal['z'] or self.uav_pos[0] < 0 or self.uav_pos[1] < 0 or self.uav_pos[2] < 0:
        #    r -= 10
        # done and reward
        # if 0 > self.uav_pos[2] or self.uav_pos[2] > 2 *  self.goal['z']:
        #    r -= 300

        if self.goal['x'] - self.goal['l']/2 < self.uav_pos[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < self.uav_pos[1] < self.goal['y'] + self.goal['l']/2:
                if self.goal['z'] - self.goal['l']/2 < self.uav_pos[2] < self.goal['z'] + self.goal['l']/2:
                    r += 10
                    self.on_goal += 1
                    if self.on_goal > 15:
                        done = True
        else:
            if self.on_goal > 0:
                self.on_goal -= 1
        if done:
            r += 1000

        #if self.uav_pos[2] >= self.goal['z']:
        #    r += 10
        #if self.uav_pos[2] < 0:
        #    r -= 100
        #if self.uav_pos[2] - 0.5 * self.goal['l'] <= self.goal['z'] <= self.uav_pos[2] + 0.5 * self.goal['l']:
        #    r += 100
        #    self.on_goal += 1
        #if self.on_goal >= 10:
        #    r += 1000
        #    done = True
        s = np.concatenate((self.uav_pos, self.uav_euler, self.uav_v, self.uav_w, [1. if self.on_goal else 0.]))
        # print 'pos: ', self.uav_pos
        return s, r, done
        #return s, done

    def d_angles_to_angular_vel(self, dtheta):
        phi = self.uav_euler[0]
        theta = self.uav_euler[1]
        trans_mat = [[1, 0, -math.sin(theta)],
                     [0, math.cos(phi), math.cos(theta) * math.sin(phi)],
                     [0, -math.sin(phi), math.cos(theta) * math.cos(phi)]]
        self.uav_w = np.dot(trans_mat, dtheta)

    def angular_vel_to_d_angles(self):
        phi = self.uav_euler[0]
        theta = self.uav_euler[1]
        trans_mat = [[1, 0, -math.sin(theta)],
                     [0, math.cos(phi), math.cos(theta) * math.sin(phi)],
                     [0, -math.sin(phi), math.cos(theta) * math.cos(phi)]]
        return np.dot(np.linalg.inv(trans_mat), self.uav_w)

    # R transfer earth to body frame
    def angles_to_R(self, thetas):
        phi = thetas[0]
        theta = thetas[1]
        psi = thetas[2]
        R = [[math.cos(psi) * math.cos(theta),
              math.cos(psi) * math.sin(theta) * math.sin(phi) - math.sin(psi) * math.cos(phi),
              math.cos(psi) * math.sin(theta) * math.cos(phi) + math.sin(psi) * math.sin(phi)],
             [math.sin(psi) * math.cos(theta),
              math.sin(psi) * math.sin(theta) * math.sin(phi) + math.cos(psi) * math.cos(phi),
              math.sin(psi) * math.sin(theta) * math.cos(phi) - math.cos(psi) * math.sin(phi)],
             [-math.sin(theta), math.cos(theta) * math.sin(phi), math.cos(theta) * math.cos(phi)]]
        return R

    def get_thrust(self, propeller_rot_speed):
        thrusts = []
        for idx in range(4):
            V = self.prop_wind_speed[idx]
            n = propeller_rot_speed[idx]
            J = V / (n) * self.propeller_size
            C_T = max(0, .12 - .07 * max(0, J) - .1 * max(0, J))
            thrusts.append(C_T * self.rho * n ** 2 * self.propeller_size ** 4)

        return np.array(thrusts)

    def get_moments(self, thrusts):
        thrust_moments = np.array(
            [(thrusts[3] - thrusts[2]) * self.l_to_rotor, (thrusts[1] - thrusts[0]) * self.l_to_rotor, 0])
        drag_moments = self.C_d * .5 * self.rho * self.uav_w * np.absolute(self.uav_w) * self.areas * np.array(
            [self.width, self.length, self.height]) * np.array([self.width, self.length, self.height])
        return thrust_moments - drag_moments

    def get_linear_forces(self, thrusts):
        gravity_forces = self.mass * self.gravity
        thrust_body_force = np.array([0., 0., sum(thrusts)])
        drag_body_force = 0.5 * self.rho * self.ground_v_to_body_v() ** 2 * self.areas * self.C_d
        # if np.isnan(drag_body_force[0]):
        # print self.uav_v, thrusts
        body_force = thrust_body_force - drag_body_force
        RT = np.transpose(self.angles_to_R(self.uav_euler))
        lin_force = np.dot(RT, body_force)
        lin_force += gravity_forces
        return lin_force

    def reset(self):
        #print "reset"
        self.uav_euler = np.array(list(self.uav_init_euler))
        self.uav_pos = np.array(list(self.uav_init_pos))  # self.goal['x'] * np.random.rand(3)
        #self.uav_init_pos = self.uav_init_pos
        self.uav_v = np.array(list(self.uav_init_v))
        self.uav_w = np.array(list(self.uav_init_w))
        self.prop_wind_speed = np.zeros(4)
        self.on_goal = 0
        #print "reset", self.uav_pos, self.uav_euler
        #dist1 = [(self.goal['x'] - self.uav_pos[0]), (self.goal['y'] - self.uav_pos[1]),
        #         (self.goal['z'] - self.uav_pos[2])]
        s = np.concatenate((self.uav_pos, self.uav_euler, self.uav_v, self.uav_w, [1. if self.on_goal else 0.]))
        return s

    def ground_v_to_body_v(self):
        body_v = np.dot(self.angles_to_R(self.uav_euler), self.uav_v)
        return body_v

    def get_prop_wind_speed(self):
        body_v = self.ground_v_to_body_v()
        phi_dot = self.uav_w[0]
        theta_dot = self.uav_w[1]
        s0 = np.array([0., 0., theta_dot * self.l_to_rotor])
        s1 = -s0
        s2 = np.array([0., 0., phi_dot * self.l_to_rotor])
        s3 = -s2
        speeds = [s0, s1, s2, s3]
        for num in range(4):
            perpen_speed = speeds[num] + body_v
            self.prop_wind_speed[num] = perpen_speed[2]
