from os import truncate
from typing import Tuple, List, Optional, Dict, Callable
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import pybullet as pb
import pybullet_data
import time
from tqdm import tqdm
import h5py
import gym
import random

class RobotArm:
    GRIPPER_CLOSED = 0.
    GRIPPER_OPENED = 1.
    def __init__(self):
        '''Robot Arm simulated in Pybullet, with support for performing top-down
        grasps within a specified workspace
        '''
        # placing robot higher above ground improves top-down grasping ability
        self._id = pb.loadURDF("nuro_arm/assets/urdf/xarm.urdf",
                               basePosition=(0, 0, 0.05),
                               flags=pb.URDF_USE_SELF_COLLISION)

        # these are hard coded based on how urdf is written
        self.arm_joint_ids = [1,2,3,4,5]
        self.gripper_joint_ids = [6,7]
        self.dummy_joint_ids = [8]
        self.finger_joint_ids = [9,10]
        self.end_effector_link_index = 11

        self.arm_joint_limits = np.array(((-2, -1.58, -2, -1.8, -2),
                                          ( 2,  1.58,  2,  2.0,  2)))
        self.gripper_joint_limits = np.array(((0.05,0.05),
                                              (1.38, 1.38)))

        # chosen to move arm out of view of camera
        self.home_arm_jpos = [0., -1.1, 1.4, 1.3, 0.]

        self.bin_1_drop_pos = [1, -1.1, -0.75, 1.3, 0.]
        self.bin_2_drop_pos = [0., -1.1, -0.2, 1.3, 0.] # TODO Change back to -0.75 when sorting
        self.bin_3_drop_pos = [-1, -1.1, -0.75, 1.3, 0.]

        # joint constraints are needed for four-bar linkage in xarm fingers
        for i in [0,1]:
            constraint = pb.createConstraint(self._id,
                                             self.gripper_joint_ids[i],
                                             self._id,
                                             self.finger_joint_ids[i],
                                             pb.JOINT_POINT2POINT,
                                             (0,0,0),
                                             (0,0,0.03),
                                             (0,0,0))
            pb.changeConstraint(constraint, maxForce=1000000)

        # reset joints in hand so that constraints are satisfied
        hand_joint_ids = self.gripper_joint_ids + self.dummy_joint_ids + self.finger_joint_ids
        hand_rest_states = [0.05, 0.05, 0.055, 0.0155, 0.031]
        [pb.resetJointState(self._id, j_id, jpos)
                 for j_id,jpos in zip(hand_joint_ids, hand_rest_states)]

        # allow finger and linkages to move freely
        pb.setJointMotorControlArray(self._id,
                                     self.dummy_joint_ids+self.finger_joint_ids,
                                     pb.POSITION_CONTROL,
                                     forces=[0,0,0])

    def move_gripper_to(self, position: List[float], theta: float, teleport: bool=False):
        '''Commands motors to move end effector to desired position, oriented
        downwards with a rotation of theta about z-axis

        Parameters
        ----------
        position
            xyz position that end effector should move toward
        theta
            rotation (in radians) of the gripper about the z-axis.

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        quat = pb.getQuaternionFromEuler((0,-np.pi,theta))

        arm_jpos = self.solve_ik(position, quat)

        if teleport:
            self.teleport_arm(arm_jpos)
            return True
        else:
            return self.move_arm_to_jpos(arm_jpos)

    def solve_ik(self,
                 pos: List[float],
                 quat: Optional[List[float]]=None,
                ) -> Tuple[List[float], Dict[str, float]]:
        '''Calculates inverse kinematics solution for a desired end effector
        position and (optionally) orientation, and returns residuals

        Hint
        ----
        To calculate residuals, you can get the pose of the end effector link using
        `pybullet.getLinkState` (but you need to set the arm joint positions first)

        Parameters
        ----------
        pos
            target xyz position of end effector
        quat
            target orientation of end effector as unit quaternion if specified.
            otherwise, ik solution ignores final orientation

        Returns
        -------
        list
            joint positions of arm that would result in desired end effector
            position and orientation. in order from base to wrist
        dict
            position and orientation residuals:
                {'position' : || pos - achieved_pos ||,
                 'orientation' : 1 - |<quat, achieved_quat>|}
        '''
        old_arm_jpos = list(zip(*pb.getJointStates(self._id, self.arm_joint_ids)))[0]

        # good initial arm jpos for ik
        [pb.resetJointState(self._id, i, jp)
            for i,jp in zip(self.arm_joint_ids, self.home_arm_jpos)]

        n_joints = pb.getNumJoints(self._id)
        all_jpos = pb.calculateInverseKinematics(self._id,
                                                 self.end_effector_link_index,
                                                 pos,
                                                 quat,
                                                 maxNumIterations=20,
                                                 jointDamping=n_joints*[0.005])
        arm_jpos = all_jpos[:len(self.arm_joint_ids)]

        self.teleport_arm(old_arm_jpos)

        return arm_jpos

    def move_arm_to_jpos(self, arm_jpos: List[float]) -> bool:
        '''Commands motors to move arm to desired joint positions

        Parameters
        ----------
        arm_jpos
            joint positions (radians) of arm joints, ordered from base to wrist

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        # cannot use setJointMotorControlArray because API does not expose
        # maxVelocity argument, which is needed for stable object manipulation
        for j_id, jpos in zip(self.arm_joint_ids, arm_jpos):
            pb.setJointMotorControl2(self._id,
                                     j_id,
                                     pb.POSITION_CONTROL,
                                     jpos,
                                     positionGain=0.2,
                                     maxVelocity=1.0)

        return self.monitor_movement(arm_jpos, self.arm_joint_ids)

    def teleport_arm(self, arm_jpos: List[float]) -> None:
        [pb.resetJointState(self._id, i, jp)
            for i,jp in zip(self.arm_joint_ids, arm_jpos)]

    def teleport_gripper(self, gripper_state: float) -> None:
        assert 0 <= gripper_state <= 1, 'Gripper state must be in range [0,1]'

        gripper_jpos = (1-gripper_state)*self.gripper_joint_limits[0] \
                       + gripper_state*self.gripper_joint_limits[1]
        [pb.resetJointState(self._id, i, jp)
            for i,jp in zip(self.gripper_joint_ids, gripper_jpos)]

        [pb.resetJointState(self._id, j_id, jpos)
                 for j_id,jpos in zip(self.hand_joint_ids, self.hand_rest_states)]

    def set_gripper_state(self, gripper_state: float) -> bool:
        '''Commands motors to move gripper to given state

        Parameters
        ----------
        gripper_state
            gripper state is a continuous number from 0. (fully closed)
            to 1. (fully open)

        Returns
        -------
        bool
            True if movement is successful, False otherwise.

        Raises
        ------
        AssertionError
            If `gripper_state` is outside the range [0,1]
        '''
        assert 0 <= gripper_state <= 1, 'Gripper state must be in range [0,1]'

        gripper_jpos = (1-gripper_state)*self.gripper_joint_limits[0] \
                       + gripper_state*self.gripper_joint_limits[1]

        pb.setJointMotorControlArray(self._id,
                                     self.gripper_joint_ids,
                                     pb.POSITION_CONTROL,
                                     gripper_jpos,
                                     positionGains=[0.2, 0.2])

        success = self.monitor_movement(gripper_jpos, self.gripper_joint_ids)
        return success

    def get_gripper_state(self):
        gripper_jpos = list(zip(*pb.getJointStates(self._id, self.gripper_joint_ids)))[0]
        gripper_state = (gripper_jpos - self.gripper_joint_limits[0]) \
                            / (self.gripper_joint_limits[1] - self.gripper_joint_limits[0])
        gripper_state = np.clip(np.mean(gripper_state), 0, 1)
        return gripper_state

    def monitor_movement(self,
                         target_jpos: List[float],
                         joint_ids: List[int],
                        ) -> bool:
        '''Monitors movement of motors to detect early stoppage or success.

        Note
        ----
        Current implementation calls `pybullet.stepSimulation`, without which the
        simulator will not move the motors.  You can avoid this by setting
        `pybullet.setRealTimeSimulation(True)` but this is usually not advised.

        Parameters
        ----------
        target_jpos
            final joint positions that motors are moving toward
        joint_ids
            the joint ids associated with each `target_jpos`, used to read out
            the joint state during movement

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        old_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
        while True:
            [pb.stepSimulation() for _ in range(10)]

            time.sleep(0.01)

            achieved_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
            if np.allclose(target_jpos, achieved_jpos, atol=1e-3):
                # success
                return True

            if np.allclose(achieved_jpos, old_jpos, atol=1e-2):
                # movement stopped
                return False
            old_jpos = achieved_jpos


class Camera:
    def __init__(self, workspace: np.ndarray, img_size: int) -> None:
        '''Camera that is mounted to view workspace from above
        Hint
        ----
        For this camera setup, it may be easiest if you use the functions
        `pybullet.computeViewMatrix` and `pybullet.computeProjectionMatrixFOV`.
        cameraUpVector should be (0,1,0)
        Parameters
        ----------
        workspace
            2d array describing extents of robot workspace that is to be viewed,
            in the format: ((min_x,min_y), (max_x, max_y))
        Attributes
        ----------
        img_size : int
            height, width of rendered image
        view_mtx : List[float]
            view matrix that is positioned to view center of workspace from above
        proj_mtx : List[float]
            proj matrix that set up to fully view workspace
        '''
        self.img_size = img_size

        cam_height = 0.25
        workspace_width = workspace[1,0] - workspace[0,0]
        fov = 2 * np.degrees(np.arctan2(workspace_width/2, cam_height))

        cx, cy = np.mean(workspace, axis=0)
        eye_pos = (cx, cy, cam_height)
        target_pos = (cx, cy, 0)
        self.view_mtx = pb.computeViewMatrix(cameraEyePosition=eye_pos,
                                             cameraTargetPosition=target_pos,
                                            cameraUpVector=(-1,0,0))
        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=fov,
                                                      aspect=1,
                                                      nearVal=0.01,
                                                      farVal=1)

    def get_rgb_image(self) -> np.ndarray:
        '''Takes rgb image
        Returns
        -------
        np.ndarray
            shape (H,W,3) with dtype=np.uint8
        '''
        rgba = pb.getCameraImage(width=self.img_size,
                                 height=self.img_size,
                                 viewMatrix=self.view_mtx,
                                 projectionMatrix=self.proj_mtx,
                                 renderer=pb.ER_TINY_RENDERER)[2]

        return rgba[...,:3]



class TopDownGraspingEnv(gym.Env):
    def __init__(self,
                 episode_length: int=1,
                 img_size: int=42,
                 render: bool=False,
                 num_blocks: int=1,
                ) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        self.client = pb.connect(pb.GUI if render else pb.DIRECT)
        pb.setPhysicsEngineParameter(numSubSteps=0,
                                     numSolverIterations=100,
                                     solverResidualThreshold=1e-7,
                                     constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI)
        pb.setGravity(0,0,-10)

        self.tex_ids = [pb.loadTexture(f) for f in glob.glob('assets/textures/*.png')]

        # create ground plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        # offset plane y-dim to place white tile under workspace
        self.plane_id = pb.loadURDF('plane.urdf', (0,-0.5,0))

        # makes collisions with plane more stable
        pb.changeDynamics(self.plane_id, -1,
                          linearDamping=0.04,
                          angularDamping=0.04,
                          restitution=0,
                          contactStiffness=3000,
                          contactDamping=100)

        self.num_blocks = num_blocks

        # add robot
        self.robot = RobotArm()

        # Add blocks
        self.objects = []
        for i in range(self.num_blocks):
            block = pb.loadURDF("nuro_arm/assets/urdf/cube.urdf")

            pb.changeDynamics(block, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)
            self.objects.append(block)
        
        self.blocks_left = len(self.objects)
        
        self.object_width = 0.02
        self.grasp_height = self.object_width/2

        self.workspace = np.array(((0.10, -0.05), # ((min_x, min_y)
                                   (0.20, 0.05))) #  (max_x, max_y))

        self.camera = Camera(self.workspace, img_size)
        self.img_size = img_size

        self.t_step = 0
        self.episode_length = episode_length

        self.observation_space = gym.spaces.Box(0, 255,
                                                shape=(img_size, img_size, 3),
                                                dtype=np.uint8)
        self.action_space = gym.spaces.Box(np.array([0,0,0]), np.array([5,img_size-1, img_size-1]), dtype=int)
    
    def reset(self) -> np.ndarray:
        '''Resets environment by randomly placing object(s)
        '''
        self.reset_objects_positions()
        self.reset_objects_colors()
        self.t_step = 0

        return self.get_obs()

    def step(self, action: np.ndarray):
        # print(action)

        assert self.action_space.contains(action)

        x,y = self._convert_from_pixel(np.array(action)[1:])
        rot = np.pi/6 * np.array(action)[0]

        success = self.perform_grasp(x, y, rot)
        self.t_step += 1

        obs = self.get_obs()
        reward = float(success)
        done = success or self.t_step >= self.episode_length
        info = {'success' : success}

        return obs, reward, done, info

    def _convert_from_pixel(self, pxy: np.ndarray) -> np.ndarray:
        xy_norm = pxy.astype(float) / self.img_size

        xy = xy_norm * np.subtract(*self.workspace[::-1]) + self.workspace[0]
        return xy

    def _convert_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy_norm = np.subtract(xy, self.workspace[0]) \
                    / np.subtract(*self.workspace[::-1])
        xy_norm = np.clip(xy_norm, 0, 1)

        #xy axis are flipped from world to image space
        pxy = self.img_size * xy_norm
        return pxy.astype(int)

    def perform_grasp(self, x: float, y: float, theta: float=0) -> bool:
        '''Perform top down grasp in the workspace.  All grasps will occur
        at a height of the center of mass of the object (i.e. object_width/2)
        Parameters
        ----------
        x
            x position of the grasp in world frame
        y
            y position of the grasp in world frame
        theta, default=0
            target rotation about z-axis of gripper during grasp
        Returns
        -------
        bool
            True if object was successfully grasped, False otherwise. It is up
            to you to decide how to determine success
        '''
        self.robot.teleport_arm(self.robot.home_arm_jpos)
        self.robot.set_gripper_state(self.robot.GRIPPER_OPENED)

        pos = np.array((x, y, self.object_width/2))

        self.robot.move_gripper_to(pos, theta, teleport=True)
        self.robot.set_gripper_state(self.robot.GRIPPER_CLOSED)
        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)

        if self.robot.get_gripper_state() < 0.05:
            # gripper closed so we know its failed grasp
            success = False
        else:
            success = True
        #     self.move_to_bin(self.robot.bin_2_drop_pos)
        #     self.blocks_left = self.blocks_left - 1

        # self.robot.teleport_arm(self.robot.home_arm_jpos)

        # # check if object is above plane
        # min_object_height = 0.2
        # obj_height = pb.getBasePositionAndOrientation(self.object_id)[0][2]
        # success = obj_height > min_object_height
        return success

    def perform_object_grasp(self, x, y, theta, object) -> bool:
        '''Perform top down grasp in the workspace.  All grasps will occur
        at a height of the center of mass of the object (i.e. object_width/2)

        Parameters
        ----------
        x
            x position of the grasp in world frame
        y
            y position of the grasp in world frame
        theta
            target rotation about z-axis of gripper during grasp

        Returns
        -------
        bool
            True if object was successfully grasped, False otherwise. It is up
            to you to decide how to determine success
        '''
        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)
        self.robot.set_gripper_state(self.robot.GRIPPER_OPENED)

        pos = np.array((x, y, self.grasp_height))
        self.robot.move_gripper_to(pos, theta)
        self.robot.set_gripper_state(self.robot.GRIPPER_CLOSED)
        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)

        if self.robot.get_gripper_state() < 0.05:
            # gripper closed so we know its failed grasp
            success = False
        else:
            success = True
        #     self.move_to_bin(self.robot.bin_2_drop_pos)
        #     self.blocks_left = self.blocks_left - 1

        # self.robot.teleport_arm(self.robot.home_arm_jpos)

        # # check if object is above plane
        # min_object_height = 0.05
        # obj_height = pb.getBasePositionAndOrientation(object)[0][2]
        # success = obj_height > min_object_height

        return success

    def move_to_bin(self, bin) -> None:
        '''Moves arm to given bin location and then drops the object in the bin.
        '''
        self.robot.move_arm_to_jpos(bin)
        self.robot.set_gripper_state(self.robot.GRIPPER_OPENED)
        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)
    def sample_random_grasp(self) -> Tuple[float, float, float]:
        '''Samples random grasp (x,y,theta) located within the workspace

        Returns
        -------
        tuple
            x, y, theta (radians)
        '''
        x,y = np.random.uniform(*self.workspace)
        theta = np.random.uniform(0, 2*np.pi)
        return x, y, theta

    def sample_expert_grasp(self, object) -> Tuple[float, float, float]:
        '''Samples expert grasp (x,y,theta) located within the workspace. An
        expert grasp can be determined using information about the state of the
        object. It should have >90% success rate if implemented correctly.

        Returns
        -------
        tuple
            x, y, theta (radians)
        '''
        obj_pos, obj_quat = pb.getBasePositionAndOrientation(object)
        x, y = obj_pos[:2]
        theta = pb.getEulerFromQuaternion(obj_quat)[2]

        return x, y, theta

    def reset_object_position(self) -> None:
        '''Places object randomly in workspace.  The x,y position should be
        within the workspace, and the rotation performed only about z-axis.
        The height of the object should be set such that it sits on the plane
        '''
        ws_padding = 0.01
        x,y = np.random.uniform(self.workspace[0]+ws_padding,
                                self.workspace[1]-ws_padding)
        theta = np.random.uniform(-np.pi/2, np.pi/2)

        pos = np.array((x,y,self.object_width/2))
        quat = pb.getQuaternionFromEuler((np.random.randint(2)*np.pi,0, theta))
        pb.resetBasePositionAndOrientation(self.object_id, pos, quat)

    def reset_object_texture(self) -> None:
        '''Randomly assigns a texture to the object.  Available textures are
        located within the `assets/textures` folder.
        '''
        tex_id = self.tex_ids[np.random.randint(len(self.tex_ids))]
        r,g,b = np.random.uniform(0.5, 1, size=3)**0.8
        pb.changeVisualShape(self.object_id, -1, -1,
                             textureUniqueId=tex_id,
                             rgbaColor=(r,g,b,1))

    def reset_objects_positions(self) -> None:
        '''Places object randomly in workspace.  The x,y position should be
        within the workspace, and the rotation performed only about z-axis.
        The height of the object should be set such that it sits on the plane
        '''
        ws_padding = 0.01
        for i in range(self.num_blocks):
            x,y = np.random.uniform(self.workspace[0]+ws_padding,
                                    self.workspace[1]-ws_padding)
            theta = np.random.uniform(-np.pi/2, np.pi/2)

            pos = np.array((x,y,self.object_width/2))
            quat = pb.getQuaternionFromEuler((np.random.randint(2)*np.pi,0, theta))
            pb.resetBasePositionAndOrientation(self.objects[i], pos, quat)

    def reset_objects_colors(self) -> List:
        '''Randomly assigns a texture to the object.  Available textures are
        located within the `nuro_arm/assets/textures` folder.
        '''
        randColors = []
        for i in range(self.num_blocks):
            colors = [(0,1,0,1), (1,0,0,1), (0,0,1,1)]
            randColor = random.choice(colors)
            randColors.append(randColor)
            pb.changeVisualShape(self.objects[i], -1, -1,
                             rgbaColor=randColor)

        return randColors

    def get_obs(self) -> np.ndarray:
        '''Takes picture using camera
        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        return self.camera.get_rgb_image()


def test_camera_placement():
    '''Use this to test your camera placement (Section 5).  If your camera is
    placed well, then the object should be fully visible regardless of where
    it is reset.

    Note
    ----
    You do not need to align the axis of the camera such that the x-axis of
    simulator is the same as x-axis of camera frame
    '''
    env = TopDownGraspingEnv(True)

    while 1:
        env.reset_object_position()
        env.get_obs()
        time.sleep(0.5)


def mock_data_collection(env):
    '''Use this to watch data collection, as per Section 6. You are not graded
    on this, but it may be a helpful sanity check that your grasp success
    determination is accurate.

    Note
    ----
    The robot may occasionally go to some random configuration in the air, this
    is due to a bad IK solution.  There are ways to avoid this, but for now you
    can ignore it.
    '''
    #env = TopDownGraspingEnv(True)

    
    success = True
    success_count = 0
    env.reset()
    objects = env.objects.copy()

    for i in range(100):
        # if (not success or env.blocks_left == 0):
        #     env.reset()
        #     objects = env.objects.copy()

        randObject = random.choice(objects)
        x,y,th = env.sample_expert_grasp(randObject)
        # objects.remove(randObject)

        success = env.perform_object_grasp(x, y, th, randObject)
        if success:
            success_count = success_count + 1
        print(success_count)
        env.reset()

    print("Success Rate: " + str(success_count))

def collect_transitions(env: TopDownGraspingEnv,
                        hdf5_file: str,
                        num_steps: int=3000,
                        policy: Optional[Callable]=None):
    if policy is None:
        policy = lambda s: env.action_space.sample()

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    s = env.reset()
    for i in tqdm(range(num_steps)):
        a = policy(s)
        sp, r, d, info = env.step(a)

        states.append(s.copy())
        actions.append(a.copy())
        rewards.append(r)
        next_states.append(sp.copy())
        dones.append(d)

        s = sp.copy()
        if d:
            s = env.reset()

    with h5py.File(hdf5_file, 'w') as hf:
        hf.create_dataset('states', data=np.array(states, dtype=np.uint8))
        hf.create_dataset('actions', data=np.array(actions, dtype=np.int8))
        hf.create_dataset('rewards', data=np.array(rewards, dtype=np.float32))
        hf.create_dataset('next_states', data=np.array(next_states, dtype=np.uint8))
        hf.create_dataset('dones', data=np.array(dones, dtype=bool))


def watch_policy(env: TopDownGraspingEnv, policy: Optional[Callable]=None):
    if policy is None:
        policy = lambda s: env.action_space.sample()

    s = env.reset()
    while 1:
        a = policy(s)
        sp, r, d, info = env.step(a)
        time.sleep(0.5)

        s = sp.copy()
        if d:
            s = env.reset()
            time.sleep(1)


if __name__ == "__main__":
    env = TopDownGraspingEnv(render=False, num_blocks=5)
    mock_data_collection(env) # Used this for demos

    #watch_policy(env)