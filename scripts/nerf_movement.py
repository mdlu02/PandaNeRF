#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

## BEGIN_SUB_TUTORIAL imports
##
## To use the Python MoveIt interfaces, we will import the `moveit_commander`_ namespace.
## This namespace provides us with a `MoveGroupCommander`_ class, a `PlanningSceneInterface`_ class,
## and a `RobotCommander`_ class. More on these below. We also import `rospy`_ and some messages that we will use:
##

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from geometry_msgs.msg import Point

import franka_gripper.msg

# Brings in the SimpleActionClient
import actionlib


#### TrackIK ####

from trac_ik_python.trac_ik import IK

#### TrackIK ####

import yaml
from yaml.loader import SafeLoader


from moveit_commander.exception import MoveItCommanderException
import moveit_commander
import actionlib
import rospy

import numpy as np
import math

import tf


try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL



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
        group_name = "panda_manipulator"
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

        self.T_matrix = np.array([[ 1.00972718, -0.03175631, -0.14718904,  0.62225599],
                             [ 0.27047383,  0.35774614,  1.17738679, -1.04287907],
                             [-0.2485964,  -0.52195157,  0.09659727,  0.32649496],
                             [ 0.,          0.,          0.,          1.,        ]])
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

        # actual_pose = np.dot(self.T_matrix, self.obj_pose)
        # print("Actual pose")
        # print(actual_pose)

        # IK_goal_pose = self.move_group.get_current_pose().pose
        # IK_goal_pose.position.x = actual_pose[0,0]
        # IK_goal_pose.position.y = actual_pose[1,0]
        # IK_goal_pose.position.z = actual_pose[2,0]
        # new_joint_state = self.get_ik_soln(IK_goal_pose)
        # self.go_to_joint_state(new_joint_state)

        # T_matrix 
        # print("Grasp pose")
        # print(self.grasp_pose)


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


    def compute_manipulability(self, joint_val, action_mean):

        #current_joint = self.move_group.get_current_joint_values()

        J = self.move_group.get_jacobian_matrix(joint_val)

        JJt = np.matmul(J,J.transpose())
        invJJt = np.linalg.inv(JJt)
        Jdet = np.linalg.det(JJt)
        w = math.sqrt(Jdet)
        #### new
        ttt = 1 / np.dot(np.dot(action_mean[None], invJJt), action_mean[None].transpose())[0][0]
        ttt = np.sqrt(ttt)
        #### new

        p = 1

        k = 1 #hyperparameter for adjust behavior near joint limits
        joint_total = 1

        for i in range(len(self.upper_joint_limit)):
            joint_total *= (joint_val[i] - self.lower_joint_limit[i])*(self.upper_joint_limit[i]-joint_val[i])/np.square(self.upper_joint_limit[i]-self.lower_joint_limit[i])

        p -= math.exp(-k*joint_total)


        return(w,ttt,np.power(w,4) * np.power(p,1) * np.power(ttt,8))



    def compute_manipulability2(self, joint_val, action_mean):

        #current_joint = self.move_group.get_current_joint_values()

        J = self.move_group.get_jacobian_matrix(joint_val)

        JJt = np.matmul(J,J.transpose())
        Jdet = np.linalg.det(JJt)
        w = math.sqrt(Jdet)

        #### new
 
        ttt = np.dot(J.transpose(), action_mean[None].transpose())[:,0]

        ttt = np.sqrt(ttt.dot(ttt))
        #### new

        p = 1

        k = 1 #hyperparameter for adjust behavior near joint limits
        joint_total = 1

        for i in range(len(self.upper_joint_limit)):
            joint_total *= (joint_val[i] - self.lower_joint_limit[i])*(self.upper_joint_limit[i]-joint_val[i])/np.square(self.upper_joint_limit[i]-self.lower_joint_limit[i])

        p -= math.exp(-k*joint_total)

        #print(w)

        return(w,ttt,np.power(w,4) * np.power(p,1) * np.power(ttt,8))


    def compute_manipulability_old(self, joint_val, action_mean):

        #current_joint = self.move_group.get_current_joint_values()

        J = self.move_group.get_jacobian_matrix(joint_val)

        JJt = np.matmul(J,J.transpose())
        Jdet = np.linalg.det(JJt)
        w = math.sqrt(Jdet)

        p = 1

        k = 1 #hyperparameter for adjust behavior near joint limits
        joint_total = 1

        for i in range(len(self.upper_joint_limit)):
            joint_total *= (joint_val[i] - self.lower_joint_limit[i])*(self.upper_joint_limit[i]-joint_val[i])/np.square(self.upper_joint_limit[i]-self.lower_joint_limit[i])

        p -= math.exp(-k*joint_total)

        return(w,p,np.power(w,4)*p)

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


    def get_best_ik_soln(self, goal, num_attempts=10000):

        best_manip = -np.inf
        best_manip2 = -np.inf ###Extra
        best_manip3 = -np.inf
        action_mean = np.mean(np.array(np.load('/home/hubo/panda_ws/src/panda_motion_planning/scripts/action_array.npy')),axis=0)

        for i in range(num_attempts):
            candidate_joints = self.get_ik_soln(goal)
            if candidate_joints == None:
                continue
            else:
                w, ttt, manipulability_score = self.compute_manipulability(candidate_joints, action_mean)
                
                if manipulability_score > best_manip:
                    best_soln = candidate_joints
                    best_manip = manipulability_score
                    best_ttt = ttt
                    best_w = w


                w2, p2, manipulability_score2 = self.compute_manipulability2(candidate_joints, action_mean)
                if manipulability_score2 > best_manip2:
                    best_soln2 = candidate_joints
                    best_manip2 = manipulability_score2
                    
                w3, p3, manipulability_score3 = self.compute_manipulability_old(candidate_joints, action_mean)
                if manipulability_score3 > best_manip3:
                    best_soln3 = candidate_joints
                    best_manip3 = manipulability_score3
                    
        print(best_soln)
        print(best_soln2)
        print(best_soln3)
        print(best_ttt)
        print(best_w)
        return best_soln

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

        ## BEGIN_SUB_TUTORIAL add_box
        ##
        ## Adding Objects to the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## First, we will create a box in the planning scene between the fingers:
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
        #box_pose.pose.position.z = 1  # above the panda_hand frame
        box_name = "box"
        scene.add_mesh(mesh_name, box_pose, '/home/hubo/Desktop/bricks_models/model_2.obj')

        ## END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def remove_mesh(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        mesh_name = self.mesh_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL remove_object
        ##
        ## Removing Objects from the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can remove the box from the world.
        scene.remove_world_object(mesh_name)

        ## **Note:** The object must be detached before we can remove it from the world
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_known=False, timeout=timeout
        )

    def add_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL add_box
        ##
        ## Adding Objects to the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## First, we will create a box in the planning scene between the fingers:
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

        ## END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
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
        group_names = self.group_names

        ## BEGIN_SUB_TUTORIAL attach_object
        ##
        ## Attaching Objects to the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## Next, we will attach the box to the Panda wrist. Manipulating objects requires the
        ## robot be able to touch them without the planning scene reporting the contact as a
        ## collision. By adding link names to the ``touch_links`` array, we are telling the
        ## planning scene to ignore collisions between those links and the box. For the Panda
        ## robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
        ## you should change this value to the name of your end effector group name.
        grasping_group = "panda_hand"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
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

        ## BEGIN_SUB_TUTORIAL detach_object
        ##
        ## Detaching Objects from the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can also detach and remove the object from the planning scene:
        scene.remove_attached_object(eef_link, name=box_name)
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )

    def remove_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL remove_object
        ##
        ## Removing Objects from the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can remove the box from the world.
        scene.remove_world_object(box_name)

        ## **Note:** The object must be detached before we can remove it from the world
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )



    def grasp_client(self, goal):
        # Creates the SimpleActionClient, passing the type of the action
        # (GraspAction) to the constructor.
        client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)

        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()


        # Sends the goal to the action server.
        client.send_goal(goal)

        # Waits for the server to finish performing the action.
        client.wait_for_result()

        # Prints out the result of executing the action
        return client.get_result()  # A GraspResult


    def homing_client(self):

        client = actionlib.SimpleActionClient('franka_gripper/homing', franka_gripper.msg.HomingAction)

        # Waits until the action server has started up and started
        # listening for goals.
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

    def calibration(self):
        print("We are starting calibration")
        print("Going to first pose")
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = -0.9252139569894603
        pose_goal.orientation.y = 0.37944258555161475
        pose_goal.orientation.z = -0.0006487140511953158
        pose_goal.orientation.w = 0.0014273163246012232
        pose_goal.position.x = 0.30807052566838294
        pose_goal.position.y = 0.000776178464688897
        pose_goal.position.z = 0.5901552576624204

        success = self.go_to_pose_goal(pose_goal)

        rospy.sleep(3.)

        cur_pose = self.move_group.get_current_pose().pose

        cur_obj_pose = self.obj_pose.copy()

        self.camera_pose_list.append(cur_obj_pose)
        self.robot_pose_list.append(np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z, 1]))
        print("Camera pos")
        print(self.camera_pose_list)

        print("Robot pos")
        print(self.robot_pose_list)

        print("Going to second pose")

        pose_goal.position.x = 0.6326607853491595
        pose_goal.position.y = -0.036455703301609764
        pose_goal.position.z = 0.3047981836140787

        success = self.go_to_pose_goal(pose_goal)

        rospy.sleep(3.)


        cur_pose = self.move_group.get_current_pose().pose

        cur_obj_pose = self.obj_pose.copy()

        self.camera_pose_list.append(cur_obj_pose)
        self.robot_pose_list.append(np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z, 1]))
        print("Camera pos")
        print(self.camera_pose_list)

        print("Robot pos")
        print(self.robot_pose_list)

        print("Going to third pose")

        pose_goal.position.x = 0.6926953257973754
        pose_goal.position.y = -0.23714038625626838
        pose_goal.position.z = 0.38583101807382925

        success = self.go_to_pose_goal(pose_goal)

        rospy.sleep(3.)


        cur_pose = self.move_group.get_current_pose().pose

        cur_obj_pose = self.obj_pose.copy()

        self.camera_pose_list.append(cur_obj_pose)
        self.robot_pose_list.append(np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z, 1]))
        print("Camera pos")
        print(self.camera_pose_list)

        print("Robot pos")
        print(self.robot_pose_list)

        print("Going to fourth pose")

        pose_goal.position.x = 0.6545550566062964
        pose_goal.position.y = -0.04140180802166248
        pose_goal.position.z = 0.48513232798299133

        success = self.go_to_pose_goal(pose_goal)

        rospy.sleep(3.)


        cur_pose = self.move_group.get_current_pose().pose

        cur_obj_pose = self.obj_pose.copy()

        self.camera_pose_list.append(cur_obj_pose)
        self.robot_pose_list.append(np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z, 1]))
        print("Camera pos")
        print(self.camera_pose_list)

        print("Robot pos")
        print(self.robot_pose_list)

        print("Going to fifth pose")

        pose_goal.position.x = 0.6006745460268843
        pose_goal.position.y = 0.26443864953610174
        pose_goal.position.z = 0.5072050606347268

        success = self.go_to_pose_goal(pose_goal)

        rospy.sleep(3.)


        cur_pose = self.move_group.get_current_pose().pose

        cur_obj_pose = self.obj_pose.copy()

        self.camera_pose_list.append(cur_obj_pose)
        self.robot_pose_list.append(np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z, 1]))
        print("Camera pos")
        print(self.camera_pose_list)

        print("Robot pos")
        print(self.robot_pose_list)

        print("Going to sixth pose")

        pose_goal.position.x = 0.6009542219022761
        pose_goal.position.y = 0.26524701834081105
        pose_goal.position.z = 0.3166234267145227

        success = self.go_to_pose_goal(pose_goal)

        rospy.sleep(3.)


        cur_pose = self.move_group.get_current_pose().pose

        cur_obj_pose = self.obj_pose.copy()

        self.camera_pose_list.append(cur_obj_pose)
        self.robot_pose_list.append(np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z, 1]))
        print("Camera pos")
        print(self.camera_pose_list)

        print("Robot pos")
        print(self.robot_pose_list)

        print("Going to seventh pose")

        pose_goal.position.x = 0.5492415948921582 -0.2 ### 0.2 added later
        pose_goal.position.y = 0.18922306861028035
        pose_goal.position.z = 0.3162127568241576

        success = self.go_to_pose_goal(pose_goal)

        rospy.sleep(3.)


        cur_pose = self.move_group.get_current_pose().pose

        cur_obj_pose = self.obj_pose.copy()

        self.camera_pose_list.append(cur_obj_pose)
        self.robot_pose_list.append(np.array([cur_pose.position.x, cur_pose.position.y, cur_pose.position.z, 1]))
        print("Camera pos")
        print(self.camera_pose_list)

        print("Robot pos")
        print(self.robot_pose_list)

        camera_array = np.array(self.camera_pose_list)
        robot_array = np.array(self.robot_pose_list)

        print("Camera pos")
        print(camera_array)

        print("Robot pos")
        print(robot_array)

        row1, residual1, rank, singularvals = np.linalg.lstsq(camera_array, robot_array[:,0][None].T, rcond = None)
        row2, residual2, rank, singularvals = np.linalg.lstsq(camera_array, robot_array[:,1][None].T, rcond = None)
        row3, residual3, rank, singularvals = np.linalg.lstsq(camera_array, robot_array[:,2][None].T, rcond = None)
        row4 = np.array([[0, 0, 0, 1]])

        print(self.T_matrix)

        self.T_matrix = np.concatenate((row1.T,row2.T,row3.T,row4), axis = 0 )

        print(self.T_matrix)
        print(residual1, residual2, residual3)


        print("Finished calibration")




# def main():
#     try:
#         print("")
#         print("----------------------------------------------------------")
#         print("Welcome to the MoveIt MoveGroup Python Interface Tutorial")
#         print("----------------------------------------------------------")
#         print("Press Ctrl-D to exit at any time")
#         print("")




        


#         # input(
#         #     "============ Press `Enter` to begin the tutorial by setting up the moveit_commander ..."
#         # )

#         tutorial = Nerf_Movement()

#         IK_goal_pose = tutorial.move_group.get_current_pose().pose

#         print("current_pose")
#         print(IK_goal_pose)




#         input(
#             "============ Press `Enter` to release ..."
#         )

#         ##Release##

#         # Creates a goal to send to the action server.
#         goal = franka_gripper.msg.GraspGoal()
#         goal.width = 0.08
#         goal.epsilon.inner = 0.005
#         goal.epsilon.outer = 0.005
#         goal.speed = 0.1
#         goal.force = 3 #CHECK BELOW BEFORE EDITING

#         # self._MIN_WIDTH = 0.0                  # [m] closed
#         # self._MAX_WIDTH = 0.08                 # [m] opened
#         # self._MIN_FORCE = 0.01                 # [N]
#         # self._MAX_FORCE = 50.0                 # [N]


#         result = tutorial.grasp_client(goal)


#         input(
#             "============ Press `Enter` to grasp ... ============="
#         )

#         # Creates a goal to send to the action server.
#         goal = franka_gripper.msg.GraspGoal()
#         goal.width = 0.01
#         goal.epsilon.inner = 0.05
#         goal.epsilon.outer = 0.05
#         goal.speed = 0.1
#         goal.force = 15 #CHECK BELOW BEFORE EDITING

#         # self._MIN_WIDTH = 0.0                  # [m] closed
#         # self._MAX_WIDTH = 0.08                 # [m] opened
#         # self._MIN_FORCE = 0.01                 # [N]
#         # self._MAX_FORCE = 50.0                 # [N]


#         result = tutorial.grasp_client(goal)







#         input(
#             "============ Press `Enter` to add the box"
#         )

#         #tutorial.add_mesh()

#         return
        
#         #input(
#         #    "============ Press `Enter` to go to initial panda_position =============="
#         #)

#         #stating_joint_state = [0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]
#         #tutorial.go_to_joint_state(stating_joint_state)
#         #tutorial.homing_client() ### Add if gripper remained closed after last execution


#         ######EXECUTE IF GRIPPERS ARE CLOSED###########

#         # input(
#         #     "============ Press `Enter` to release ..."
#         # )

#         # ##Release##

#         # # Creates a goal to send to the action server.
#         # goal = franka_gripper.msg.GraspGoal()
#         # goal.width = 0.08
#         # goal.epsilon.inner = 0.005
#         # goal.epsilon.outer = 0.005
#         # goal.speed = 0.1
#         # goal.force = 3 #CHECK BELOW BEFORE EDITING

#         # # self._MIN_WIDTH = 0.0                  # [m] closed
#         # # self._MAX_WIDTH = 0.08                 # [m] opened
#         # # self._MIN_FORCE = 0.01                 # [N]
#         # # self._MAX_FORCE = 50.0                 # [N]


#         # result = tutorial.grasp_client(goal)

#         #############################################################################################

#         input(
#             "============ Press `Enter` to execute a movement using a joint state goal ..."
#         )

#         #EDIT desired joint values here

#         input_joints=[0, -tau / 8, 0, -tau / 4, 0, tau / 6, 0]
#         input_joints=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
#         action = np.array([0, 0, 0.4, 0, 0, 0])
#         w_tulu, tt_tulu, s_tulu = tutorial.compute_manipulability(input_joints,action)
#         w_paper, tt_paper, s_paper = tutorial.compute_manipulability2(input_joints,action)
#         w_old, p, s_old = tutorial.compute_manipulability_old(input_joints,action)
#         print('Tulu')
#         print(w_tulu, tt_tulu, s_tulu)
#         print('Paper')
#         print(w_paper, tt_paper, s_paper)
#         print('Old')
#         print(w_old, p, s_old)


#         #tutorial.go_to_joint_state(input_joints)


#         ##################### EDIT #######################

#         quaternion = tf.transformations.quaternion_from_euler(np.pi, 0, -np.pi/4) # (roll, pitch, yaw)   Practical to use euler

#         pose_goal = geometry_msgs.msg.Pose()
#         pose_goal.orientation.x = quaternion[0]
#         pose_goal.orientation.y = quaternion[1]
#         pose_goal.orientation.z = quaternion[2]
#         pose_goal.orientation.w = quaternion[3]
#         pose_goal.position.x = 0.4
#         pose_goal.position.y = 0.1
#         pose_goal.position.z = 0.5

#         new_joint_state = tutorial.get_best_ik_soln(pose_goal)


#         # joint_val = tutorial.move_group.get_current_joint_values()

#         # J = tutorial.move_group.get_jacobian_matrix(joint_val)



        

#         # JJt = np.matmul(J,J.transpose())
#         # Jdet = np.linalg.det(JJt)
#         # w = math.sqrt(Jdet)

#         # m = len(J)

#         # d = np.power(2*pi, m/2) / (2 * 4 * 6)


#         # U_matrix, s_vector, Vh_matrix = np.linalg.svd(J, full_matrices = True)
#         # S_matrix = np.diag(np.square(s_vector))


#         # print(np.asmatrix(JJt))
#         # E = np.dot(np.dot(U_matrix, S_matrix),U_matrix.transpose())
#         # print(np.asmatrix(E))
#         # pseudoJJt = np.linalg.pinv(JJt)
#         # print(np.asmatrix(pseudoJJt))
#         # volume = np.linalg.det(E)
#         # volume2 = np.linalg.det(pseudoJJt)
#         # print(w)
#         # print(volume)
#         # print(np.sqrt(volume))
#         # print(volume2)
#         # print(np.sqrt(volume2))
#         # print(w)
#         # print(d)

#         # action_list = np.load('/home/hubo/panda_ws/src/panda_motion_planning/scripts/action_array.npy')
#         # print(len(action_list))
#         # b = action_list[0]
#         # print(np.array(b[None]))
#         # print(np.array(b[None]).transpose())

#         # k = 1
#         # for s in s_vector:
#         #     k *= s

#         # print(w)
#         # print(k)






#         #return

#         ##################### EDIT #######################

#         #############################################################################################
        

#         ### CIPS ####

#         input(
#             "============ Press `Enter` to go to IK solved pose ================"
#         )

#         IK_goal_pose = tutorial.move_group.get_current_pose().pose
#         new_joint_state = tutorial.get_best_ik_soln(IK_goal_pose)
#         tutorial.go_to_joint_state(new_joint_state)

#         #############################################################################################


#         input("============ Press `Enter` to execute a movement using a pose goal ...")

#         #EDIT POSE HERE

#         quaternion = tf.transformations.quaternion_from_euler(np.pi, 0, -np.pi/4) # (roll, pitch, yaw)   Practical to use euler

#         pose_goal = geometry_msgs.msg.Pose()
#         pose_goal.orientation.x = quaternion[0]
#         pose_goal.orientation.y = quaternion[1]
#         pose_goal.orientation.z = quaternion[2]
#         pose_goal.orientation.w = quaternion[3]
#         pose_goal.position.x = 0.3 #0.6
#         pose_goal.position.y = 0 #0.1
#         pose_goal.position.z = 0.4#0.2

#         IK_goal_pose = tutorial.move_group.get_current_pose().pose
#         IK_goal_pose.position.x += 0.1
#         new_joint_state = tutorial.get_best_ik_soln(IK_goal_pose)

#         #new_joint_state = tutorial.get_best_ik_soln(pose_goal)
#         tutorial.go_to_joint_state(new_joint_state)
#         #tutorial.go_to_pose_goal(pose_goal)


#         input(
#             "============ Press `Enter` to grasp ... ============="
#         )

#         # Creates a goal to send to the action server.
#         goal = franka_gripper.msg.GraspGoal()
#         goal.width = 0.01
#         goal.epsilon.inner = 0.05
#         goal.epsilon.outer = 0.05
#         goal.speed = 0.1
#         goal.force = 15 #CHECK BELOW BEFORE EDITING

#         # self._MIN_WIDTH = 0.0                  # [m] closed
#         # self._MAX_WIDTH = 0.08                 # [m] opened
#         # self._MIN_FORCE = 0.01                 # [N]
#         # self._MAX_FORCE = 50.0                 # [N]


#         result = tutorial.grasp_client(goal)



#         input(
#             "============ Press `Enter` to go to new pose ... ============="
#         )

#         #EDIT POSE HERE

#         quaternion = tf.transformations.quaternion_from_euler(np.pi, 0, -np.pi/4) # (roll, pitch, yaw)   Practical to use euler

#         pose_goal = geometry_msgs.msg.Pose()
#         pose_goal.orientation.x = quaternion[0]
#         pose_goal.orientation.y = quaternion[1]
#         pose_goal.orientation.z = quaternion[2]
#         pose_goal.orientation.w = quaternion[3]
#         pose_goal.position.x = 0.6
#         pose_goal.position.y = 0.1
#         pose_goal.position.z = 0.4

#         new_joint_state = tutorial.get_best_ik_soln(pose_goal)
#         tutorial.go_to_joint_state(new_joint_state)



#         input(
#             "============ Press `Enter` to release ..."
#         )

#         ##Release##

#         # Creates a goal to send to the action server.
#         goal = franka_gripper.msg.GraspGoal()
#         goal.width = 0.08
#         goal.epsilon.inner = 0.005
#         goal.epsilon.outer = 0.005
#         goal.speed = 0.1
#         goal.force = 3 #CHECK BELOW BEFORE EDITING

#         # self._MIN_WIDTH = 0.0                  # [m] closed
#         # self._MAX_WIDTH = 0.08                 # [m] opened
#         # self._MIN_FORCE = 0.01                 # [N]
#         # self._MAX_FORCE = 50.0                 # [N]


#         result = tutorial.grasp_client(goal)


#         # input("============ Press `Enter` to plan and display a Cartesian path ...")
#         # cartesian_plan, fraction = tutorial.plan_cartesian_path()

#         # input(
#         #     "============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ..."
#         # )
#         # tutorial.display_trajectory(cartesian_plan)

#         # input("============ Press `Enter` to execute a saved path ...")
#         # tutorial.execute_plan(cartesian_plan)

#         # input("============ Press `Enter` to add a box to the planning scene ...")
#         # tutorial.add_box()

#         # #input("============ Press `Enter` to attach a Box to the Panda robot ...")
#         # #tutorial.attach_box()

#         # input(
#         #     "============ Press `Enter` to plan and execute a path with an attached collision object ..."
#         # )
#         # cartesian_plan, fraction = tutorial.plan_cartesian_path(scale=-1)
#         # tutorial.execute_plan(cartesian_plan)

#         # # input("============ Press `Enter` to detach the box from the Panda robot ...")
#         # # tutorial.detach_box()

#         # input(
#         #     "============ Press `Enter` to remove the box from the planning scene ..."
#         # )
#         # tutorial.remove_box()

#         print("============ Python tutorial demo complete!")
#     except rospy.ROSInterruptException:
#         return
#     except KeyboardInterrupt:
#         return


if __name__ == "__main__":
    #main()
    try:
        tutorial = Nerf_Movement()
        stating_joint_state = [0.017295376760307065, -1.4765418056583965, -0.049415543378438447, -3.0051199601376184, -0.013422825147092919, 1.5766404801091545, 0.8293549973573249]
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = 1
        pose_goal.orientation.y = 0
        pose_goal.orientation.z = 0
        pose_goal.orientation.w = 0

        pose_goal.position.x = 0.5167311544024981
        pose_goal.position.y = -0.0898657740611424
        pose_goal.position.z = -0.2052250712667316
        # pose_goal.position.x = 0.564
        # pose_goal.position.y = 0.0000
        # pose_goal.position.z = 0.083


        #success = self.go_to_pose_goal(pose_goal)
        breakpoint()
        sys.exit()
        tutorial.add_mesh(0.475 + 0.2664, -0.086)#0.73)
        #tutorial.open_gripper()
        input("============ Ready to start??? ================")
        tutorial.go_to_joint_state(stating_joint_state)
        #breakpoint()
        #sys.exit()
        #tutorial.calibration()
        #breakpoint()
        #sys.exit()
        #input("============ Remove mesh ================")
        #tutorial.remove_mesh()


        
        input("============ Ready to follow??? ================")
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = -0.9231885858846031
        pose_goal.orientation.y = 0.3815944210387882
        pose_goal.orientation.z = -0.0350003179578283
        pose_goal.orientation.w = 0.0297239039715496
        pose_goal.position.x = 0.504275
        pose_goal.position.y = -0.20624
        pose_goal.position.z = 0.17790193959865177
        tutorial.go_to_pose_goal(pose_goal)
        tutorial.close_gripper()

        input("Open gripper")

        tutorial.open_gripper()
        tutorial.go_to_joint_state(stating_joint_state)

        sys.exit()
        breakpoint()
        sys.exit()
        # tutorial.open_gripper()
        tutorial.close_gripper()
        tutorial.calibration()


        input("============ Ready to follow??? ================")

        tutorial.open_gripper()



        for i in range(5):
            input("============ Execute next ================")
            print(tutorial.obj_pose)
            input("============ Start ================")
            goal_pose = np.dot(tutorial.T_matrix, tutorial.obj_pose)
            print("Goal pose")
            print(goal_pose)

            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.orientation.x = -0.9252139569894603
            pose_goal.orientation.y = 0.37944258555161475
            pose_goal.orientation.z = -0.0006487140511953158
            pose_goal.orientation.w = 0.0014273163246012232
            pose_goal.position.x = goal_pose[0]
            pose_goal.position.y = goal_pose[1]
            pose_goal.position.z = goal_pose[2]


            success = tutorial.go_to_pose_goal(pose_goal)


        # IK_goal_pose = tutorial.move_group.get_current_pose().pose

        # print("current_pose")
        # print(IK_goal_pose)

        #input("============ Press `Enter` to go to next pose")

        #tutorial.open_gripper()
        #tutorial.close_gripper()
        # goal = franka_gripper.msg.GraspGoal()
        # goal.width = 0.08
        # goal.epsilon.inner = 0.005
        # goal.epsilon.outer = 0.005
        # goal.speed = 0.1
        # goal.force = 3 #CHECK BELOW BEFORE EDITING

        # # self._MIN_WIDTH = 0.0                  # [m] closed
        # # self._MAX_WIDTH = 0.08                 # [m] opened
        # # self._MIN_FORCE = 0.01                 # [N]
        # # self._MAX_FORCE = 50.0                 # [N]


        # result = tutorial.grasp_client(goal)
        #tutorial.add_mesh()
        # rospy.sleep(10.)
        # print("hebelererkltjerkltjklertjljerkltj")
        # IK_goal_pose = tutorial.move_group.get_current_pose().pose

        # print("current_pose")
        # print(IK_goal_pose)
        #rospy.spin()
    except rospy.ROSInterruptException:
        pass

## BEGIN_TUTORIAL
## .. _moveit_commander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/namespacemoveit__commander.html
##
## .. _MoveGroupCommander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html
##
## .. _RobotCommander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1robot_1_1RobotCommander.html
##
## .. _PlanningSceneInterface:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1planning__scene__interface_1_1PlanningSceneInterface.html
##
## .. _DisplayTrajectory:
##    http://docs.ros.org/noetic/api/moveit_msgs/html/msg/DisplayTrajectory.html
##
## .. _RobotTrajectory:
##    http://docs.ros.org/noetic/api/moveit_msgs/html/msg/RobotTrajectory.html
##
## .. _rospy:
##    http://docs.ros.org/noetic/api/rospy/html/
## CALL_SUB_TUTORIAL imports
## CALL_SUB_TUTORIAL setup
## CALL_SUB_TUTORIAL basic_info
## CALL_SUB_TUTORIAL plan_to_joint_state
## CALL_SUB_TUTORIAL plan_to_pose
## CALL_SUB_TUTORIAL plan_cartesian_path
## CALL_SUB_TUTORIAL display_trajectory
## CALL_SUB_TUTORIAL execute_plan
## CALL_SUB_TUTORIAL add_box
## CALL_SUB_TUTORIAL wait_for_scene_update
## CALL_SUB_TUTORIAL attach_object
## CALL_SUB_TUTORIAL detach_object
## CALL_SUB_TUTORIAL remove_object
## END_TUTORIAL
