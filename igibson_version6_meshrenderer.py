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
import cv2
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.utils.assets_utils import get_scene_path
#########################################################################################################
########################################### ACTION PROC #################################################
#########################################################################################################
def TakeAction(direction,rob_env,XY,current_index_position):
    A=int(current_index_position[0])
    B=int(current_index_position[1])
    if(direction=='UP'):
        ##assigning the co-ordinates using index positions
        position=[XY[A],XY[B],0]
        rob_env.robots[0].set_position_orientation(position,[0,0,0,1])
        print(position)
        if(A!=(len(XY)-1)):
            final_index_position=[int(A+1),int(B)]
            final_target_position=[XY[A+1],XY[B],0]
            index_position,final_position=planner(current_index_position,final_index_position,XY)
            return index_position,final_position
        else:
            print("grid limit in X+ direction reached")
            return current_index_position,position
    elif(direction=='DOWN'):
        position=[XY[A],XY[B],0]
        rob_env.robots[0].set_position_orientation(position,[0,0,1,0])
        print(position)
        if(A!=0):
            final_index_position=[A-1,B]
            final_target_position=[XY[A-1],XY[B],0]
            index_position,final_position=planner(current_index_position,final_index_position,XY)
            return index_position,final_position
        else:
            print("grid limit in X- direction reached")
            return current_index_position,position
    elif(direction=='RIGHT'):
        position=[XY[A],XY[B],0]
        rob_env.robots[0].set_position_orientation(position,[0,0,-0.707,0.707])
        print(position)
        if(B!=0):
            final_index_position=[A,B-1]
            final_target_position=[XY[A],XY[B-1],0]
            index_position,final_position=planner(current_index_position,final_index_position,XY)
            return index_position,final_position
        else:
            print("grid limit in Y- direction reached")
            return current_index_position,position
    elif(direction=='LEFT'):
        position=[XY[A],XY[B],0]
        rob_env.robots[0].set_position_orientation(position,[0,0,1,1])
        print(position)
        if(B!=(len(XY)-1)):
            final_index_position=[A,B+1]
            final_target_position=[XY[A],XY[B+1],0]
            index_position,final_position=planner(current_index_position,final_index_position,XY)
            return index_position,final_position
        else:
            print("grid limit in Y+ direction reached")
            return current_index_position,position

#####################################################################################################################
####################################################PLANNER##########################################################
#####################################################################################################################
def planner(current_index_position,final_index_position,XY):
    final_target_position=[XY[int(final_index_position[0])],XY[int(final_index_position[1])],0]
    current_position=[XY[int(current_index_position[0])],XY[int(current_index_position[1])],0]
    plan = None
    itr = 0
    flag=0
    while plan is None and itr < 10:
        plan = motion_planner.plan_base_motion(final_target_position)
        if(plan==None):
            flag=1
            break
        #print(plan)
        itr += 1
    if(flag==0):
        motion_planner.dry_run_base_plan(plan)
        final_position=final_target_position
        index_position=final_index_position
        return index_position,final_position
    elif(flag==1):
        final_position=current_position
        index_position=current_index_position
        return index_position,final_position
####################################################################################################################
#####################################################REWARD#########################################################
####################################################################################################################
def ReturnReward(position,goal):
    if(position[0]<goal[0]+0.1 and position[0] > goal[0]-0.1 and position[0]<goal[0]+0.1 and position[0] > goal[0]-0.1 ):
        return 1
    else:
        return 0
###################################################################################################################
###################################################PRINT GRID######################################################
###################################################################################################################
def printGrid(grid):
    for i in range(0,len(grid)):
        print(grid[i])



#################################################################################################
#################################Environment Definitions#########################################
#################################################################################################
config_filename = parse_config(os.path.join(gibson2.example_config_path, 'fetch_motion_planning.yaml'))
settings = MeshRendererSettings(enable_shadow=False, msaa=False)
nav_env = iGibsonEnv(config_file=config_filename,
                                  mode='iggui',
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)

camera_pose = np.array([1, 1, 0.5])
view_direction = np.array([0, -1, -1])
nav_env.simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
nav_env.simulator.renderer.set_fov(90)
#while True:
#        frame = nav_env.simulator.renderer.render(modes=('rgb'))
#        cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
#        cv2.waitKey()
#        break
#nav_env.simulator.load()
#navrenderer.load_object(model_path)

#exit()

motion_planner = MotionPlanningWrapper(nav_env)
goal=[1,-0.2,0]
n=21
grid=[[0]*n for _ in range(0,n)]
printGrid(grid)
m=10
XY=[]
for i in range(0,21):
    o=(i-m)/5
    XY.append(o)
print(XY)
initial_index=[15,10]

class CustomGridEnv(gym.Env):
    def __init__(self,nav_env,goal,grid,initial_index,XY):
        super(CustomGridEnv,self).__init__()
        self.current_X=int(initial_index[0])
        self.current_Y=int(initial_index[1])
        position=[XY[self.current_X],XY[self.current_Y],0]
        nav_env.reset()
        nav_env.robots[0].set_position_orientation(position,[0,0,0,1])
        self.nav_env=nav_env
        self.motion_planner=MotionPlanningWrapper(self.nav_env)
        self.goal=goal
        self.grid=grid
        self.grid[self.current_X][self.current_Y]=1
        #printGrid(self.grid)
        self.XY=XY
    def step(self,action):
        current_index_position=[self.current_X,self.current_Y]
        self.grid[self.current_X][self.current_Y]=0
        final_index_position,final_target_position=TakeAction(action,self.nav_env,self.XY,current_index_position)
        index_position,position=planner(final_index_position,final_target_position,self.XY)
        self.current_X=index_position[0]
        self.current_Y=index_position[1]
        self.grid[self.current_X][self.current_Y]=1
        obs=self.nav_env.get_state()
        reward=ReturnReward(position,self.goal)
        info={}
        return obs,position,reward,info,self.grid
        #print("step")
    def reset(self):
        nav_env.reset()
        nav_env.robots[0].set_position_orientation([0,0,0],[0,0,0,1])
        #print("reset")


RobotEnv=CustomGridEnv(nav_env,goal,grid,initial_index,XY)
obs,position,Reward,info,grid=RobotEnv.step('RIGHT')
print("goal position is")
print(goal)
print("current robot position is")
print(position)
print("Reward")
print(Reward)
print("The robot position in the grid")
printGrid(grid)
obs,position,Reward,info,grid=RobotEnv.step('DOWN')
print("goal position is")
print(goal)
print("current robot position is")
print(position)
print("Reward")
print(Reward)
print("The robot position in the grid")
printGrid(grid)
#cv2.imshow(256*obs.get('rgb'))
#picture=obs.get('rgb')
#print(len(picture))
#print(len(picture[0]))
#print(len(picture[0][0]))

#cv2.imshow("image",picture)
#cv2.waitKey()


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







