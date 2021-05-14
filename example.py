from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
#from gibson2.objects import URDFObject
from gibson2.simulator import Simulator
import gibson2
import os
import pybullet as p
import numpy as np
import pybullet_data
import time
from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.robots.fetch_robot import Fetch
from gibson2.utils.utils import parse_config
from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.render.profiler import Profiler
from IPython import embed
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
from gibson2.envs.igibson_env import iGibsonEnv


###########################script starts#########################################
#################################################################################

###start the gui###
#p.connect(p.GUI)

######################initialize gravity and time###############################
#p.setGravity(0,0,-9.8)
#p.setTimeStep(1./360.)


############################load a scene#######################################
############To load a Interactive Indoor scene using a simulator###############
s = Simulator(mode='gui', image_width=512,
                  image_height=512)
scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False)
s.import_ig_scene(scene)


################################################################################
##we keep importing things to the simulator s##################
################################################################################
#scene=InteractiveIndoorScene()
#scene=StadiumScene()
#scene.load()

#######################loading a floor#########################################
#print(pybullet_data.getDataPath())
#floor=os.path.join(pybullet_data.getDataPath(),"mjcf/ground_plane.xml")
#p.loadMJCF(floor)

######################loading objects##########################################

#cabinet_0007 = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
#cabinet_0004 = os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

############################################loading carpet ###################################

carpet= os.path.join(gibson2.ig_dataset_path,'objects/carpet/carpet_0/carpet_0.urdf')

#obj1 = ArticulatedObject(filename=cabinet_0007)
#obj1.load()
#obj1.set_position([0, 0, 0.5])

#obj2 = ArticulatedObject(filename=cabinet_0004)
#obj2.load()
#obj2.set_position([0, 0, 2])

#obj3 = YCBObject('003_cracker_box')
#obj3.load()
#obj3.set_position_orientation([0, 0, 1.2], [0, 0, 0, 1])
#obj3.set_position([0,0,1.2])

obj4= ArticulatedObject(filename=carpet)
s.import_object(obj4)
obj4.set_position([0,1,0])


np.random.seed(0)
for _ in range(10):
    pt = scene.get_random_point_by_room_type('living_room')[1]
    print('random point in living_room', pt)


##############################################adding a fetch robot###################################################
config = parse_config(os.path.join(gibson2.example_config_path, 'fetch_motion_planning.yaml'))

settings = MeshRendererSettings(enable_shadow=False, msaa=False)
#turtlebot = Turtlebot(config)
#robot=turtlebot
#position=[1,1,0]
fetchbot = Fetch(config)
s.import_robot(fetchbot)
fetchbot.set_position([0,1,0])

##################################################################################################################
##tried changing robot control to position and torque but failed,from the documentation it seems turtle bot only##
############################################supports joint velocity###############################################
##################################################################################################################
#robot.load()
#robot.set_position(position)
#robot.robot_specific_reset()
#robot.keep_still()
#print(robot.action_dim)
#print(robot.control)
############################################adding keyboard functionality#####################################
#print(p.getKeyboardEvents())
##############################################start the simulation############################################
for i in range(360):
    #p.stepSimulation()
    s.step()
    #time.sleep(1./360.)

#for i in range(1):
    #action = np.random.uniform(-1, 1, robot.action_dim)
    #print(action)
    #action=np.array([0,0.5])
    #turtlebot.apply_action(action)
    #print(robot.get_angular_velocity())
    #robot.move_forward(forward=0.5)
    #p.stepSimulation()
    #s.step()
    #robot.keep_still()
    #time.sleep(1./360.)
for i in range(1000):
    #with Profiler('Simulator step'):
    action=np.random.uniform(-1,1,fetchbot.action_dim)
    fetchbot.apply_action(action)
    s.step()
    rgb = s.renderer.render_robot_cameras(modes=('rgb'))

for i in range(3600):
    s.step()
    #time.sleep(1./360.) 


###end the gui###
#p.disconnect()
s.disconnect()
