import os
import random
import sys
import time
import traci.constants as tc
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "C:/Sumo/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "./jatek.sumocfg", "--start"]
traci.start(sumoCmd)
print("Starting SUMO")
# addSubscriptionFilterLanes
j = 0
egoID = None
speed = []
pos = []
head = []  # tomb inint
t = time.time()
lanes = [-1, 0, 1]
# traci.vehicle.subscribeContext("ego", tc.CMD_GET_VEHICLE_VARIABLE, 0.0, [tc.VAR_SPEED])
# traci.vehicle.addSubscriptionFilterLanes(lanes, noOpposite=True, downstreamDist=100, upstreamDist=50)
traci.vehicletype.setColor("car2", (255, 0, 0, 0))

while True:

    IDsOfVehicles = traci.vehicle.getIDList()
    NumOfVehicles = len(IDsOfVehicles)
    if NumOfVehicles > 20 and egoID is None:
        egoID = random.choice(IDsOfVehicles)
        traci.vehicle.setLaneChangeMode(egoID, 0)
        traci.vehicle.setColor(egoID, (125, 0, 255, 0))
        traci.vehicle.setSpeedMode(egoID, 0)
        # prevy=traci.vehicle.getPosition(egoID)[1]
        # edge_lane = traci.vehicle.getLaneID(egoID).split("_")
        # traci.vehicle.moveTo(egoID, traci.vehicle.getLaneID(egoID), 0.2)
        # traci._vehicle.setParameter()
    traci.simulationStep()

    #  if NumOfVehicles > 0 and egoID in IDsOfVehicles:
    # temp_pos = traci.vehicle.getPosition(egoID)
    # traci.vehicle.moveTo(egoID,traci.vehicle.getLaneID(egoID),0.2)
    #       for veh_id in IDsOfVehicles:
    #   if veh_id != egoID:
    #      prev_pos=traci.vehicle.getPosition(veh_id)
    #     prev_posx=prev_pos[0]-temp_pos[0]
    #    edge_lane=traci.vehicle.getLaneID(veh_id)
    # traci.vehicle.setParameter(veh_id,"position","%s, %s" %(prev_posx,prev_pos[1]))
    #   if len(edge_lane) ==2:
    #       traci.vehicle.moveTo(veh_id,edge_lane,prev_posx)
