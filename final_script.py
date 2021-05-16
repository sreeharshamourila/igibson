from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
from gibson2.objects.articulated_object import ArticulatedObject
import gibson2
import argparse
import numpy as np
import os
from gibson2.utils.utils import parse_config
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
carpet= os.path.join(gibson2.ig_dataset_path,'objects/carpet/carpet_0/carpet_0.urdf')
obj4= ArticulatedObject(filename=carpet)


##########################################################################################
#####################################Initializing the positions###########################
##########################################################################################
state = nav_env.reset()
nav_env.robots[0].set_position_orientation([1,0,0],[0,0,0,1])
#nav_env.simulator(mode='iggui', image_width=512,image_height=512)
nav_env.simulator.import_object(obj4)
obj4.set_position([1,0,0])
#for i in range(1000):
nav_env.simulator.step()


#####################################################################
################controlling with keyboard############################
#for i in range(0,4):
print("please input")
a=input()
print("input is")
print(a)
if(a=='1'):
    position=nav_env.robots[0].get_position()
    print(position)
    position[2]=0
    print(position+step_x)
    plan = None
    itr = 0
    while plan is None and itr < 10:
        plan = motion_planner.plan_base_motion(position+step_x)
        print(plan)
        itr += 1
    motion_planner.dry_run_base_plan(plan)
    position=nav_env.robots[0].get_position()
    print(position)











