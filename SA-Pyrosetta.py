import random as rnd
import math
import copy

from rosetta import *
from pyrosetta import *
from pyrosetta.teaching import *
import rosetta.protocols.rigid as rigid_moves
from rosetta.protocols.rigid import *
from pyrosetta import PyMOLMover

init()
pose1=pose_from_pdb("1AJX.pdb")
pose2=Pose()
pose2.assign(pose1)
setup_foldtree(pose1,"AB_X",Vector1([1]))
setup_foldtree(pose2,"AB_X",Vector1([1]))
jump_num = 1

BodyPosition = rosetta.numeric

sfxn = create_score_function('ligand')
pymover = PyMOLMover()
pymover.apply(pose1)


#-----------initial conditions-----------------
initial_temperature = 100
cooling = 0.5                
number_variables = 8
upper_bounds = [1.0,1.0,1.0,30.0,1.0,1.0,1.0,5.5]
lower_bounds = [-1.0,-1.0,-1.0,-30.0,-1.0,-1.0,-1.0,-5.5]
#-----------------------------------------------
#x=10,387/y=18,119/z=8,484
#---------------variables------------------------
xrot=0.0
yrot=0.0
zrot=0.0
angle_magnitude=2
axisRot = BodyPosition.xyzVector_double_t(xrot,yrot,zrot)
xtrans=0.0
ytrans=0.0
ztrans=0.0
step_size=1
axisTrans = BodyPosition.xyzVector_double_t(xtrans,ytrans,ztrans)
#-----------------------------------------------


#-------------initial movement------------------
axis = BodyPosition.xyzVector_double_t(1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(-1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(-1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(-1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(-1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(-1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(-1.0,0.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()

axis = BodyPosition.xyzVector_double_t(0.0,1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,-1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,-1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,-1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,-1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,-1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,-1.0,0.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()

axis = BodyPosition.xyzVector_double_t(0.0,0.0,1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,-1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,-1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,-1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,-1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,-1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
axis = BodyPosition.xyzVector_double_t(0.0,0.0,-1.0)
trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
trans_mover.trans_axis(axis)
trans_mover.step_size(5.5)
trans_mover.apply(pose2)
pymover.apply(pose2)
pose2=pose1.clone()
#-----------------------------------------------

def objective_function(X):

    global pose1,pose2
    
    anxisRot = BodyPosition.xyzVector_double_t(X[0],X[1],X[2])
    spin_mover = rigid_moves.RigidBodyDeterministicSpinMover()
    spin_mover.rb_jump(jump_num)
    spin_mover.spin_axis(anxisRot)
    spin_mover.rot_center(pyrosetta.rosetta.core.pose.get_center_of_mass(pose2))
    spin_mover.angle_magnitude(X[3])
    spin_mover.apply(pose2)
    
    axisTrans = BodyPosition.xyzVector_double_t(X[4],X[5],X[6])
    trans_mover = rigid_moves.RigidBodyTransMover(pose2,jump_num)
    trans_mover.trans_axis(axisTrans)
    trans_mover.step_size(X[7])
    trans_mover.apply(pose2)
  
    energy = sfxn(pose2)
    return energy


initial_solution=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for v in range(number_variables):
   initial_solution[v]= rnd.uniform(lower_bounds[v], upper_bounds[v])
current_solution=copy.copy(initial_solution)
best_solution=copy.copy(initial_solution)
n=1
best_fitness = objective_function(best_solution)
current_temperature = copy.copy(initial_temperature)
no_attempts = 10
current_fitness=0
for i in range(20):
    for j in range(no_attempts):
        for k in range(number_variables):
            #current_solution[k] = rnd.uniform(lower_bounds[k], upper_bounds[k])
            current_solution[k] = copy.copy(best_solution[k])+0.1*(rnd.uniform(lower_bounds[k],upper_bounds[k]))
            current_solution[k] = max(min(current_solution[k],upper_bounds[k]),lower_bounds[k])
        current_fitness = objective_function(current_solution)
        E = abs(current_fitness - best_fitness)
        if i==0 and j==0:
            EA=copy.copy(E)
        print('current_fitness: {}, best_fitness: {}'.format(current_fitness, best_fitness))
        if current_fitness > best_fitness:
            p = math.exp(-E/(EA*current_temperature))
            # Make a decision to Accept the worse solution or not
            if rnd.random() < p:
                accept = True #This worse solution is accepted
            else:
                accept = False #This worse solution is not accepted
        else:
            accept = True # Accept better soluctions
        if accept == True:
            
            best_solution = copy.copy(current_solution) # Update the best solution
            best_fitness = copy.copy(current_fitness)
            pymover.apply(pose2)
            pose2=pose1.clone()
            n=n + 1 # Count the solution accepted
            EA = (EA*(n-1) + E)/n
        if accept == False:
            pose2=pose1.clone()
        #pymover.apply(pose2)
    print('interation: {}, best_solution: {}, best_fitness: {}'.format(i, best_solution, best_fitness))
    
    #print(sfxn(pose1))
    #sfxn.show(pose1)
    #pymover.apply(pose2)
    # Cooling the temperature
    current_temperature = current_temperature*cooling