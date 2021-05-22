from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
from gibson2.objects.articulated_object import ArticulatedObject
import gibson2
import argparse
import numpy as np
import os
from gibson2.utils.utils import parse_config
import gym
from gym import spaces
#########################################################################################################
########################################### ACTION PROC #################################################
#########################################################################################################
def TakeAction(direction,rob_env):
    step_x=[0.2,0,0]
    step_y=[0,0.2,0]
    if(direction=='UP'):
        position=rob_env.robots[0].get_position()
        rob_env.robots[0].set_position_orientation(position,[0,0,0,1])
        print(position)
        #position[2]=0
        #print(position+step_x)
        final_position=position+step_x
        return final_position
    elif(direction=='DOWN'):
        position=rob_env.robots[0].get_position()
        rob_env.robots[0].set_position_orientation(position,[0,0,1,0])
        print(position)
        #position[2]=0
        #print(position-step_x)
        final_position=position-step_x
        return final_position
    elif(direction=='RIGHT'):
        position=rob_env.robots[0].get_position()
        rob_env.robots[0].set_position_orientation(position,[0,0,-0.707,0.707])
        print(position)
        #position[2]=0
        #print(position-step_y)
        final_position=position-step_y
        return final_position
    elif(direction=='LEFT'):
        position=rob_env.robots[0].get_position()
        rob_env.robots[0].set_position_orientation(position,[0,0,1,1])
        print(position)
        #position[2]=0
        #print(position+step_y)
        final_position=position+step_y
        return final_position
#####################################################################################################################
####################################################PLANNER##########################################################
#####################################################################################################################
def planner(final_position):
    plan = None
    itr = 0
    while plan is None and itr < 10:
        plan = motion_planner.plan_base_motion(final_position)
        #print(plan)
        itr += 1
    motion_planner.dry_run_base_plan(plan)
    position=nav_env.robots[0].get_position()
    #print(position)
    return position
####################################################################################################################
#####################################################REWARD#########################################################
####################################################################################################################
def ReturnReward(position,goal):
    if(position[0]<goal[0]+0.1 and position[0] > goal[0]-0.1 and position[0]<goal[0]+0.1 and position[0] > goal[0]-0.1 ):
        return 1
    else:
        return 0
#################################################################################################
#################################Environment Definitions#########################################
#################################################################################################
config_filename = parse_config(os.path.join(gibson2.example_config_path, 'fetch_motion_planning.yaml'))
nav_env = iGibsonEnv(config_file=config_filename,
                                  mode='iggui',
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)
motion_planner = MotionPlanningWrapper(nav_env)
goal=[1,-0.2,0]


class CustomGridEnv(gym.Env):
    def __init__(self,nav_env,motion_planner,goal):
        super(CustomGridEnv,self).__init__()
        self.nav_env=nav_env
        self.motion_planner=motion_planner
        self.goal=goal
        nav_env.reset()
        nav_env.robots[0].set_position_orientation([1,0,0],[0,0,0,1]) 
        #print("initialzed")
    def step(self,action):
        final_position=TakeAction(action,self.nav_env)
        position=planner(final_position)
        reward=ReturnReward(position,self.goal)
        return position,reward
        #print("step")
    def reset(self):
        nav_env.reset()
        nav_env.robots[0].set_position_orientation([1,0,0],[0,0,0,1])
        #print("reset")


RobotEnv=CustomGridEnv(nav_env,motion_planner,goal)
position,Reward=RobotEnv.step('RIGHT')
print("goal position is")
print(goal)
print("current robot position is")
print(position)
print("Reward")
print(Reward)
position,Reward=RobotEnv.step('DOWN')
print("goal position is")
print(goal)
print("current robot position is")
print(position)
print("Reward")
print(Reward)

exit()
########################action step for every keyboard action#########################
step_x=[0.2,0,0]
step_y=[0,0.2,0]


#################################defining config and env files########################
#config_filename = os.path.join(gibson2.root_path, 'test', 'test_house_occupancy_grid.yaml')
config_filename = parse_config(os.path.join(gibson2.example_config_path, 'fetch_motion_planning.yaml'))
nav_env = iGibsonEnv(config_file=config_filename,
                                  mode='iggui',
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)
motion_planner = MotionPlanningWrapper(nav_env)

#################################Declaring an object######################################
#carpet= os.path.join(gibson2.ig_dataset_path,'objects/carpet/carpet_0/carpet_0.urdf')
#carpet= os.path.join(gibson2.ig_dataset_path,'objects/rugs/RUGS/RUGS.urdf')
#carpet= os.path.join(gibson2.ig_dataset_path,'objects/articulated_object/carpet1/carpet1.urdf')
#carpet= os.path.join(gibson2.ig_dataset_path,'objects/new_object/Carpet2/Carpet2.urdf')
#carpet= os.path.join(gibson2.ig_dataset_path,'objects/new_object/Carpet3/Carpet3.urdf')
#carpet= os.path.join(gibson2.ig_dataset_path,'objects/new_object/Carpet4/Carpet4.urdf')
#carpet= os.path.join(gibson2.ig_dataset_path,'objects/new_object/Carpet5/Carpet5.urdf')
carpet= os.path.join(gibson2.ig_dataset_path,'objects/new_object/Carpet6/Carpet6.urdf')


#carpet2="carpet.urdf"
obj4= ArticulatedObject(filename=carpet)



##########################################################################################
#####################################Initializing the positions###########################
##########################################################################################
state = nav_env.reset()
nav_env.robots[0].set_position_orientation([1,0,0],[0,0,0,1])
#nav_env.simulator(mode='iggui', image_width=512,image_height=512)
nav_env.simulator.import_object(obj4)
obj4.set_position_orientation([-1,1,0],[0,0,0,1])
for i in range(1000):
    nav_env.simulator.step()

#exit()
#####################################################################
################controlling with keyboard############################
for i in range(1,5):
#    print("please input")
#    a=input()
#    print("input is")
    a=str(3)
    #a='1'
    print(a)
    final_position=0
    if(a=='1'):
        final_position=TakeAction('UP',nav_env)
    if(a=='2'):
        final_position=TakeAction('DOWN',nav_env)
    if(a=='3'):
        final_position=TakeAction('RIGHT',nav_env)
    if(a=='4'):
        final_position=TakeAction('LEFT',nav_env)
    planner(final_position)
    for i in range(100):
        nav_env.simulator.step()

exit()
###################################################################################
###################################openAi gym env##################################
###################################################################################
#for j in range(10):
#nav_env.reset()
for i in range(1000):
    #with Profiler('Environment action step'):
    action = nav_env.action_space.sample()
    #print(action)
    state, reward, done, info = nav_env.step(action)
    #print(state)
    #print(done)
    print(info)
    if done:
                    #logging.info(
                    #    "Episode finished after {} timesteps".format(i + 1))
        break
nav_env.close()







