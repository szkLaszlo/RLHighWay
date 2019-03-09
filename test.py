import gym
import gym_highway
from gym_highway.modell.model import Model
#env = gym.make('EPHighWay-v0')
import matplotlib.pyplot as plt
import math
#env.reset()

m=Model()
print("Lanes: ", m.lanes)
m.warmup(False)
m.search_ego_vehicle()
m.render(True)
for i in range(1000):
    if m.ego_vehicle.vx<130/3.6:
        acc=1
    else:
        acc=0
    m.one_step([0.0003 * math.sin(i / 10), acc])
    #m.onestep([0, acc])
    #print(round(m.egovehicle.vx*3.6,2))
    state=m.generate_state_for_ego()
    print("FL:"+str(round(state[0],0))+ "   FE"+str(round(state[2],0))+"   FR"+str(round(state[4],0)) )
    print("RL:"+str(round(state[6],0))+ "   RE"+str(round(state[8],0))+"   RR"+str(round(state[10],0)) )
    print("SL:"+str(round(state[12],0))+ "  SR"+str(round(state[13],0)))
    print("y:" + str(round(state[14], 0)) + "   th" + str(round(state[15], 5)) + "   v" + str(round(state[16], 0)))
    print("------------------" )
    m.render(close=True)

exit(10)
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())