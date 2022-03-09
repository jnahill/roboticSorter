from typing import Tuple, List, Optional, Dict
import time
import glob
import numpy as np
import pybullet as pb
import pybullet_data
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

    def move_gripper_to(self, position: List[float], theta: float):
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
        arm_jpos, _ = self.solve_ik(position, quat)

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
        n_joints = pb.getNumJoints(self._id)
        all_jpos = pb.calculateInverseKinematics(self._id,
                                                 self.end_effector_link_index,
                                                 pos,
                                                 quat,
                                                 maxNumIterations=20,
                                                 jointDamping=n_joints*[0.005])
        arm_jpos = all_jpos[:len(self.arm_joint_ids)]

        # teleport arm to check acheived pos and orientation
        old_arm_jpos = list(zip(*pb.getJointStates(self._id, self.arm_joint_ids)))[0]
        [pb.resetJointState(self._id, i, jp) for i,jp in zip(self.arm_joint_ids, arm_jpos)]
        achieved_pos, achieved_quat = pb.getLinkState(self._id, self.end_effector_link_index)[:2]
        [pb.resetJointState(self._id, i, jp) for i,jp in zip(self.arm_joint_ids, old_arm_jpos)]

        residuals = {'position' : np.linalg.norm(np.subtract(pos, achieved_pos)),
                     'orientation' : 1 - np.abs(np.dot(quat, achieved_quat))}

        return arm_jpos, residuals

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
                                     maxVelocity=0.8)

        return self.monitor_movement(arm_jpos, self.arm_joint_ids)

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

            if np.allclose(achieved_jpos, old_jpos, atol=1e-3):
                # movement stopped
                return False
            old_jpos = achieved_jpos


class Camera:
    def __init__(self, workspace: np.ndarray) -> None:
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
        img_width : int
            width of rendered image
        img_height : int
            height of rendered image
        view_mtx : List[float]
            view matrix that is positioned to view center of workspace from above
        proj_mtx : List[float]
            proj matrix that set up to fully view workspace
        '''
        self.img_width = 100
        self.img_height = 100

        cx, cy = np.mean(workspace, axis=0)
        eye_pos = (cx, cy, 0.25)
        target_pos = (cx, cy, 0)
        self.view_mtx = pb.computeViewMatrix(cameraEyePosition=eye_pos,
                                             cameraTargetPosition=target_pos,
                                            cameraUpVector=(0,1,0))
        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=42,
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
        rgba = pb.getCameraImage(width=self.img_width,
                                 height=self.img_height,
                                 viewMatrix=self.view_mtx,
                                 projectionMatrix=self.proj_mtx,
                                 renderer=pb.ER_TINY_RENDERER)[2]

        return rgba[...,:3]


class TopDownGraspingEnv:
    def __init__(self, render: bool=True) -> None:
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

        # add robot
        self.robot = RobotArm()

        # add object
        self.object_id = pb.loadURDF("nuro_arm/assets/urdf/cube.urdf")
        pb.changeDynamics(self.object_id, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)
        self.object_width = 0.02

        self.workspace = np.array(((0.10, -0.05), # ((min_x, min_y)
                                   (0.20, 0.05))) #  (max_x, max_y))
        self.grasp_height = self.object_width/2
        if render:
            self.draw_workspace()

        # add camera
        self.camera = Camera(self.workspace)

    def draw_workspace(self) -> None:
        '''This is just for visualization purposes, to help you with the object
        resetting.  Must be in GUI mode, otherwise error occurs

        Note
        ----
        Pybullet debug lines only show up in GUI mode so they won't help you
        with camera placement.
        '''
        corner_ids = ((0,0), (0,1), (1,1), (1,0), (0,0))
        for i in range(4):
            start = (*self.workspace[corner_ids[i],[0,1]], 0.)
            end = (*self.workspace[corner_ids[i+1],[0,1]], 0.)
            pb.addUserDebugLine(start, end, (0,0,0), 3)

    def perform_grasp(self, x, y, theta) -> bool:
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

        # check if object is above plane
        min_object_height = 0.05
        obj_height = pb.getBasePositionAndOrientation(self.object_id)[0][2]
        success = obj_height > min_object_height

        return success

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

    def sample_expert_grasp(self) -> Tuple[float, float, float]:
        '''Samples expert grasp (x,y,theta) located within the workspace. An
        expert grasp can be determined using information about the state of the
        object. It should have >90% success rate if implemented correctly.

        Returns
        -------
        tuple
            x, y, theta (radians)
        '''
        obj_pos, obj_quat = pb.getBasePositionAndOrientation(self.object_id)
        x, y = obj_pos[:2]
        theta = pb.getEulerFromQuaternion(obj_quat)[2]

        return x, y, theta

    def reset_object_position(self) -> None:
        '''Places object randomly in workspace.  The x,y position should be
        within the workspace, and the rotation performed only about z-axis.
        The height of the object should be set such that it sits on the plane
        '''
        x,y = np.random.uniform(self.workspace[0], self.workspace[1])
        theta = np.random.uniform(0, 2*np.pi)
        pos = np.array((x,y,self.object_width/2))
        quat = pb.getQuaternionFromEuler((0,0, theta))
        pb.resetBasePositionAndOrientation(self.object_id, pos, quat)

    def reset_object_texture(self) -> None:
        '''Randomly assigns a texture to the object.  Available textures are
        located within the `nuro_arm/assets/textures` folder.
        '''
        colors = [(0,1,0,1), (1,0,0,1), (0,0,1,1)]
        randColor = random.choice(colors)
        pb.changeVisualShape(self.object_id, -1, -1,
                             rgbaColor=randColor)

    def take_picture(self) -> np.ndarray:
        '''Takes picture using camera

        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        return self.camera.get_rgb_image()


def check_workspace_reachability():
    '''Use this to test your solve_ik implementation.  If it is working properly
    then you should see a green semi circle that extends up to the edge of the
    workspace.  Outside of this range, the robot is not able to perform a top
    down grasp.
    '''
    env = TopDownGraspingEnv(True)

    top_down_quat = pb.getQuaternionFromEuler((0,-np.pi,0))

    # perform scan over x,y positions
    for x in np.linspace(0.1, 0.3, num=20):
        for y in np.linspace(-0.15, 0.15, num=20):
            pos = np.array((x, y, env.grasp_height))
            _, residuals = env.robot.solve_ik(pos, top_down_quat)

            if residuals['position'] < 1e-2 and residuals['orientation'] < 1e-3:
                color = (0,1,0) # green means its feasible
            else:
                color = (1,0,0) # red means not feasible

            pb.addUserDebugLine(pos-0.001, pos+0.001, color, 10)

    time.sleep(50)


def test_move_gripper_to():
    '''Use this to test your implementation of `RobotArm.move_gripper_to`.
    If it is working correctly, then the gripper should move to a position, then slowly
    rotate its gripper, before returning to the home position.
    '''
    env = TopDownGraspingEnv(True)

    while 1:
        env.robot.move_arm_to_jpos(env.robot.home_arm_jpos)

        position = (0.16, 0., 0.01)
        for theta in np.linspace(-np.pi/4, np.pi/4, num=6, endpoint=True):
            env.robot.move_gripper_to(position, theta)
            time.sleep(0.1)

        time.sleep(0.5)


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
        env.take_picture()
        time.sleep(0.5)


def mock_data_collection():
    '''Use this to watch data collection, as per Section 6. You are not graded
    on this, but it may be a helpful sanity check that your grasp success
    determination is accurate.

    Note
    ----
    The robot may occasionally go to some random configuration in the air, this
    is due to a bad IK solution.  There are ways to avoid this, but for now you
    can ignore it.
    '''
    env = TopDownGraspingEnv(True)

    while True:
        env.reset_object_position()
        env.reset_object_texture()
        img = env.take_picture()

        # if np.random.random() > 0.5:
        #     x,y,th = env.sample_expert_grasp()
        # else:
        #     x,y,th = env.sample_random_grasp()
        x,y,th = env.sample_expert_grasp()

        success = env.perform_grasp(x, y, th)
        print("SUCCESS!" if success else "failure")


if __name__ == "__main__":
    mock_data_collection()