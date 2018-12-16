import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pos=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pos, init_velocities, init_angle_velocities, runtime)
        self.start_pos = self.sim.pose[:3]
        self.action_repeat = 3

        self.state_size = self.action_repeat * (6 + 3 + 3)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.x, self.y, self.z = self.target_pos

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        penalty = 0
        x, y, z = current_position = self.sim.pose[:3]

        # penalty for euler angles, we want the takeoff to be stable
        penalty += abs(self.sim.pose[3:6]).sum()

        # penalty constant
        pen = 0.0002

        # penalty for distance from target
        penalty += (x-self.x)**2
        penalty += (y-self.y)**2
        penalty += 10*(z-self.z)**2

        # link velocity to residual distance
        penalty += abs(abs(current_position-self.target_pos).sum() - abs(self.sim.v).sum())
        
        # Segment distance and velocity for z-axis only
        penalty += (abs(z - z.self) - self.sim.v[2])**2

        # Euclidean distance
        distance = np.sqrt((x-self.x)**2 + (y-self.y)**2 + (z-self.z)**2)

        # extra reward for flying near the target
        if distance < 100:
            reward += 1000
        elif distance < 10:
            reward += 10000

        # constant reward for flying
        reward += 100
        return reward - penalty*pen



        # current_pos = self.sim.pose[:3]
        # current_vel = self.sim.v
        # # distance = np.linalg.norm(current_pos - self.target_pos)
        # min_pos = np.array([.0,.0,.0])
        # max_pos = np.array([1000,1000,1000]
        # norm_cur_pos = (current_pos - min_pos)/(max_pos - min_pos)
        # norm_target_pos = (self.target_pos - min_pos) / (max_pos - min_pos)

        # x, y, z = norm_cur_pos
        # self.x, self.y, self.z = self.target_pos
        # v_x, v_y, v_z = current_vel

        # distance = np.sqrt((x-self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        # dist_reward = 1 - (distance)**0.4
        # vel_discount = (1 - max(v_z, 0.01)**(1.0 / max(distance, 0.01)))

        # reward = vel_discount * dist_reward
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = np.tanh(1 - 0.003*(distance))

        # reward = np.tanh(1 - 0.003*(
        #     abs(current_pos - self.target_pos))).sum()

        # reward = 1.0 / sum((self.sim.pose[:3] - self.target_pos)**2)
        # reward = 1 - sum(np.abs(current_pos - self.target_pos))
        #sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            state = self.current_state()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def current_state(self):
        """The state contains information about current position, velocity and angular velocity"""
        state = np.concatenate([np.array(self.sim.pose),
                                np.array(self.sim.v),
                                np.array(self.sim.angular_v)])
        return state


    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
