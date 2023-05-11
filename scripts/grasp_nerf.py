
#!/usr/bin/env python

from __future__ import print_function
import tf
import cv2
import sys
import math
import copy
import yaml
import rospy
import actionlib
import subprocess
import numpy as np
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import franka_gripper.msg
import pyrealsense2 as rs
from six.moves import input
from std_msgs.msg import String
from yaml.loader import SafeLoader
from geometry_msgs.msg import Point
from trac_ik_python.trac_ik import IK
from scipy.spatial.transform import Rotation as R
from moveit_commander.conversions import pose_to_list
from moveit_commander.exception import MoveItCommanderException

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class Nerf_Movement(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(Nerf_Movement, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("Nerf_Movement", anonymous=True)


        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )


        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL


        # Misc variables
        self.box_name = ""
        self.mesh_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.upper_joint_limit, self.lower_joint_limit, self.vel_limit, self.torque_limit = self.read_robot_limits()
        self.ik_solver = IK("panda_link0", "panda_hand_tcp")
        self.obj_pose = np.ones((4))
        self.camera_pose_list = []
        self.robot_pose_list = []

        rospy.Subscriber('/center_of_the_ball', Point, self.pose_callback)

        print('Upper joint limit')
        print(self.upper_joint_limit)
        print('Lower joint limit')
        print(self.lower_joint_limit)
        print('Velocity limit')
        print(self.vel_limit)
        print('Torque limit')
        print(self.torque_limit)


    def pose_callback(self, point):
        
        self.obj_pose[0] = point.x 
        self.obj_pose[1] = point.y 
        self.obj_pose[2] = point.z


    def read_robot_limits(self):

        with open('/home/hubo/panda_ws/src/franka_ros/franka_description/robots/panda/joint_limits.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        upper_joint_limit = [data['joint1']['limit']['upper'], data['joint2']['limit']['upper'], data['joint3']['limit']['upper'], data['joint4']['limit']['upper'],
                            data['joint5']['limit']['upper'], data['joint6']['limit']['upper'], data['joint7']['limit']['upper']]
        lower_joint_limit = [data['joint1']['limit']['lower'], data['joint2']['limit']['lower'], data['joint3']['limit']['lower'], data['joint4']['limit']['lower'],
                            data['joint5']['limit']['lower'], data['joint6']['limit']['lower'], data['joint7']['limit']['lower']]
        velocity_limit = [data['joint1']['limit']['velocity'], data['joint2']['limit']['velocity'], data['joint3']['limit']['velocity'], data['joint4']['limit']['velocity'],
                            data['joint5']['limit']['velocity'], data['joint6']['limit']['velocity'], data['joint7']['limit']['velocity']]
        torque_limit = [data['joint1']['limit']['effort'], data['joint2']['limit']['effort'], data['joint3']['limit']['effort'], data['joint4']['limit']['effort'],
                            data['joint5']['limit']['effort'], data['joint6']['limit']['effort'], data['joint7']['limit']['effort']]
        
        return upper_joint_limit, lower_joint_limit, velocity_limit, torque_limit


    def get_ik_soln(self, goal, seed_state=None):
        thres = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # random seed 
        if seed_state is None: 
            seed_state = np.random.uniform(low=self.lower_joint_limit, 
                                           high=self.upper_joint_limit)

        new_joint_state = self.ik_solver.get_ik(seed_state,
                            goal.position.x, goal.position.y, goal.position.z,  # X, Y, Z
                            goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w)  # QX, QY, QZ, QW
        
        if new_joint_state == None:
            return None
        else:
            return list(new_joint_state)



    def go_to_joint_state(self, input_joints):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = input_joints[0]#0
        joint_goal[1] = input_joints[1]#-tau / 8
        joint_goal[2] = input_joints[2]#0
        joint_goal[3] = input_joints[3]#-tau / 4
        joint_goal[4] = input_joints[4]#0
        joint_goal[5] = input_joints[5]#tau / 6  # 1/6 of a turn
        joint_goal[6] = input_joints[6]#0

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self, pose_goal):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:


        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def plan_cartesian_path(self, scale=1):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        ##
        waypoints = []

        wpose = move_group.get_current_pose().pose
        wpose.position.z -= scale * 0.1  # First move up (z)
        wpose.position.y += scale * 0.2  # and sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

        ## END_SUB_TUTORIAL

    def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        ## BEGIN_SUB_TUTORIAL display_trajectory
        ##
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)

        ## END_SUB_TUTORIAL

    def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        move_group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL

    def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4
    ):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL wait_for_scene_update
        ##
        ## Ensuring Collision Updates Are Received
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node was just created (https://github.com/ros/ros_comm/issues/176),
        ## or dies before actually publishing the scene update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed.
        ## To avoid waiting for scene updates like this at all, initialize the
        ## planning scene interface with  ``synchronous = True``.
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL


    def add_mesh(self,x,y, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        mesh_name = self.mesh_name
        scene = self.scene
        box_pose = geometry_msgs.msg.PoseStamped()


        box_pose.header.frame_id = "panda_hand"
        box_pose.header.frame_id = "panda_link0"
        box_pose.pose.position.x = x#1.1 #old 0.75
        box_pose.pose.position.y = y
        box_pose.pose.position.z = -0.83 # old -0.5
        box_pose.pose.orientation.x = 0
        box_pose.pose.orientation.y = 0
        box_pose.pose.orientation.z = -0.70710678
        box_pose.pose.orientation.w = 0.70710678
        box_name = "box"
        scene.add_mesh(mesh_name, box_pose, '/home/hubo/Desktop/bricks_models/model_2.obj')
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def remove_mesh(self, timeout=4):
        mesh_name = self.mesh_name
        scene = self.scene
        scene.remove_world_object(mesh_name)
        return self.wait_for_state_update(box_is_known=False, timeout=timeout
        )

    def add_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "panda_hand"
        box_pose.header.frame_id = "panda_link0"
        box_pose.pose.position.x = 1.0
        box_pose.pose.position.y = 1.0
        box_pose.pose.position.z = 1.0
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = 0.11  # above the panda_hand frame
        box_name = "box"
        scene.add_box(box_name, box_pose, size=(0.075, 0.075, 0.075))
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def attach_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link
        # group_names = self.group_names
        grasping_group = "panda_hand"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)
        return self.wait_for_state_update(
            box_is_attached=True, box_is_known=False, timeout=timeout
        )

    def detach_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene
        eef_link = self.eef_link
        scene.remove_attached_object(eef_link, name=box_name)
        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )

    def remove_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene
        scene.remove_world_object(box_name)
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )



    def grasp_client(self, goal):
        # Creates the SimpleActionClient, passing the type of the action
        # (GraspAction) to the constructor.
        client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)
        client.wait_for_server()
        client.send_goal(goal)
        client.wait_for_result()
        return client.get_result()  # A GraspResult


    def homing_client(self):

        client = actionlib.SimpleActionClient('franka_gripper/homing', franka_gripper.msg.HomingAction)
        client.wait_for_server()

        goal = franka_gripper.msg.HomingGoal()
        client.send_goal(goal)

        client.wait_for_result()

        # Prints out the result of executing the action
        return client.get_result()  # A GraspResult

    def open_gripper(self):

        goal = franka_gripper.msg.GraspGoal()
        goal.width = 0.08
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.005
        goal.speed = 0.1
        goal.force = 3 #CHECK BELOW BEFORE EDITING

        # self._MIN_WIDTH = 0.0                  # [m] closed
        # self._MAX_WIDTH = 0.08                 # [m] opened
        # self._MIN_FORCE = 0.01                 # [N]
        # self._MAX_FORCE = 50.0                 # [N]


        result = self.grasp_client(goal)

    def close_gripper(self):

        goal = franka_gripper.msg.GraspGoal()
        goal.width = 0.01
        goal.epsilon.inner = 0.05
        goal.epsilon.outer = 0.05
        goal.speed = 0.1
        goal.force = 15 #CHECK BELOW BEFORE EDITING

        # self._MIN_WIDTH = 0.0                  # [m] closed
        # self._MAX_WIDTH = 0.08                 # [m] opened
        # self._MIN_FORCE = 0.01                 # [N]
        # self._MAX_FORCE = 50.0                 # [N]


        result = self.grasp_client(goal)
        return result


def get_quaternion(right_hem, i, tup, center):
    aiming_position = np.array(center)
    current_camera_position = np.array([tup[0], tup[1], tup[2]])
    norm = np.linalg.norm(aiming_position - current_camera_position) 
    fromTtoC = (aiming_position- current_camera_position) / norm
    up_vector = [aiming_position[0] - current_camera_position[0], 
                aiming_position[1] - current_camera_position[1], 
                0.0]
    norm_up_vector = np.linalg.norm(up_vector)
    up_vector = up_vector / norm_up_vector
    cross_right = np.cross(fromTtoC, up_vector)
    norm = np.linalg.norm(cross_right)
    cameraRight = cross_right / norm # check if float
    cameraUp = np.cross(cameraRight, fromTtoC)
    
    # rotation = np.array([[cameraRight[0], cameraUp[0], fromTtoC[0], 0],
    #                      [cameraRight[1], cameraUp[1], fromTtoC[1], 0],
    #                      [cameraRight[2], cameraUp[2], fromTtoC[2], 0],
    #                      [0, 0, 0, 1]])

    # rotation = np.array([[fromTtoC[0], cameraLeft[0], cameraUp[0], 0],
    #                      [fromTtoC[1], cameraLeft[1], cameraUp[1], 0],
    #                      [fromTtoC[2], cameraLeft[2], cameraUp[2], 0],
    #                      [0.0, 0, 0, 1]])

    # rotation = np.array([[cameraUp[0], cameraLeft[0], fromTtoC[0], 0],
    #                      [cameraUp[1], cameraLeft[1], fromTtoC[1], 0],
    #                      [cameraUp[2], cameraLeft[2], fromTtoC[2], 0],
    #                      [0.0, 0, 0, 1]])

    temp_rotation = np.array([[cameraUp[0], cameraRight[0], fromTtoC[0]],
                         [cameraUp[1], cameraRight[1], fromTtoC[1]],
                         [cameraUp[2], cameraRight[2], fromTtoC[2]]
                         ])

    # rotation = np.array([[fromTtoC[0], cameraLeft[0], cameraUp[0]],
    #                      [fromTtoC[1], cameraLeft[1], cameraUp[1]],
    #                      [fromTtoC[2], cameraLeft[2], cameraUp[2]]
    #                      ])


    ## specify angle or use rotation matrix to compute angle.
    if right_hem:
        temp_rotation = R.from_euler('xyz', [0, -35, (360/20 * i)], degrees=True)
    else:
        temp_rotation = R.from_euler('xyz', [0, -35, -(360/20 * i)], degrees=True)
    temp_rotation = temp_rotation.as_matrix()
    default = R.from_euler('xyz', [-180, 0, -45], degrees=True)
    default = default.as_matrix()
    
    # default = np.hstack((default, np.array([0,0,0]).reshape(3,1)))
    # default = np.vstack((default, np.array([0,0,0,1]).reshape(1,4)))
    # print(default)
    # r = R.from_matrix(temp_rotation)
    # print(r.as_euler('xyz', degrees=True))
    # quaternion = tf.transformations.quaternion_from_matrix( default@rotation ) 
    full_rot = temp_rotation @ default
    r = R.from_matrix(full_rot)
    quaternion = r.as_quat()

    return quaternion

def circular_trajectory(tutorial, pose_goal):
    tutorial.go_to_pose_goal(pose_goal)
    sampled_points = 20 ## number of points to sample
    z = 0.51 ## height relative to base 
    radius = 0.22 ## radius of circular path
    angle_step = 2 * np.pi / sampled_points
    # center_x = 0.305710315 
    center_x = 0.37 ## x coordinate of view point
    center_y = 0.0 ## y coordinate of view point
    points = []
    right_hem = True

    ## Two loops, one for first semi-circle and one for the other. We need two
    ## since we need to resent panda's joint angles half way through to avoid 
    ## hitting joint angle limits.

    for i in range(sampled_points / 2 + 1):
        angle = angle_step * i
        x = center_x + radius * (-math.cos(angle))
        y = center_y + radius * (-math.sin(angle))

        quat = get_quaternion(right_hem, i, (x,y,z), [center_x, center_y, z])
        points.append((
                x, y, z, quat[0], quat[1], quat[2], quat[3]
            ))


    for i in range(1, sampled_points / 2):
        angle = angle_step * i
        x = center_x + radius * (-math.cos(angle))
        y = center_y + radius * (math.sin(angle))

        quat = get_quaternion(not right_hem, i, (x,y,z), [center_x, center_y, z])
        points.append((
                x, y, z, quat[0], quat[1], quat[2], quat[3]
            ))
    return points


def capture_image(i, pipeline):
        
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # if not depth_frame or not color_frame:
    #     continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        print('taking image')
        cv2.imwrite('./images/image{:03d}.png'.format(i), resized_color_image)
        cv2.imwrite('./depths/image{:03d}.pfm'.format(i), depth_colormap)
    else:
        print('taking image')
        cv2.imwrite('./images/image{:03d}.png'.format(i), color_image)
        cv2.imwrite('./depths/image{:03d}.pfm'.format(i), depth_colormap)

def upload_image_to_s3(image_path, bucket_name):
        try:
            # Build the AWS CLI command
            aws_command = f"aws s3 cp {image_path} s3://{bucket_name}/"
            
            # Run the command using subprocess
            process = subprocess.run(aws_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Print the output on successful execution
            if process.returncode == 0:
                print(f"Successfully uploaded {image_path} to {bucket_name}")
            else:
                print(f"Error uploading {image_path} to {bucket_name}: {process.stderr.decode('utf-8')}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error uploading {image_path} to {bucket_name}: {e.output.decode('utf-8')}")

def main():
    print("")
    print("----------------------------------------------------------")
    print("Welcome to the MoveIt MoveGroup Python Interface Tutorial")
    print("----------------------------------------------------------")
    print("Press Ctrl-D to exit at any time")
    print("")


    try:
        tutorial = Nerf_Movement()
        stating_joint_state = [0.03202341903301707, 0.45900370514601985, 0.0743635250858064, -0.8394780465249851, 0.01546591704007652, 0.7776030993991428, 0.8335337665490805]
        pose_goal = geometry_msgs.msg.Pose()
        stating_joint_state = [0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]
        reached = False
        print('Moving to start')
        while not reached:
            reached = tutorial.go_to_joint_state(stating_joint_state)
        # starting position
        pose_goal.orientation.x = -0.9235819691804623
        pose_goal.orientation.y = 0.38339598084673
        pose_goal.orientation.z = -0.0012122719194721824
        pose_goal.orientation.w = 0.0015487001345217454
        pose_goal.position.x = 0.3078280377829998
        pose_goal.position.y = 0.0010744939372069422
        pose_goal.position.z = 0.5898451756657014

        points = circular_trajectory(tutorial, pose_goal)

        # move to start position before starting circulating
        print('Starting path')
        reached = False
        while not reached:
            reached = tutorial.go_to_pose_goal(pose_goal)

        print('Configureing camera...')
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            pipeline.stop()
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        pipeline.start(config)
        capture_image(20, pipeline)
        for i, point in enumerate(points):
            print(f'Moving to position {i}')
            pose_goal.position.x = point[0]
            pose_goal.position.y = point[1]
            pose_goal.position.z = point[2]
            pose_goal.orientation.x = point[3]
            pose_goal.orientation.y = point[4]
            pose_goal.orientation.z = point[5]
            pose_goal.orientation.w = point[6]
            # print(pose_goal)
            reached = False
            while not reached:
                reached = tutorial.go_to_pose_goal(pose_goal)
            capture_image(i, pipeline)
            upload_image_to_s3('./images/image{:03d}.png'.format(i), 'pandanerf')
                
            if i == 10:
                stating_joint_state = [0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]
                reached = False
                print('Moving to start')
                while not reached:
                    reached = tutorial.go_to_joint_state(stating_joint_state)

        stating_joint_state = [0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]
        reached = False
        

        # Stop streaming
        pipeline.stop()   
        
        upload_image_to_s3('./images/image20.png', 'pandanerf')

        print('Moving to start')
        
        while not reached:
            reached = tutorial.go_to_joint_state(stating_joint_state)
        return
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    
    try:
        main()

    except rospy.ROSInterruptException:
        pass

