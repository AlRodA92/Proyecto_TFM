import math
import numpy as np
from torch import detach

def reward_task_objetive(obj,task_reward,dist_prev,time):

    # Initialization
    reward_objetive = 0
  
    if obj.env.map_name == "Multi_task_6m1M1Gh1scv_vs_8m1M1Gh":
        x_agent_init = 15.5
        y_agent_init = 3.0
    elif obj.env.map_name == "Multi_task_6m1M1Gh1scv_vs_12m1M1Gh":
        x_agent_init = 15.5
        y_agent_init = 3.0     
    elif obj.env.map_name == "Multi_task_6m1M1Gh1scv":
        x_agent_init = 15.5
        y_agent_init = 3.0 

    # Get target point
    pos = obj.env.get_target_point(0)
    x_target = pos.x
    y_target = pos.y

    # Get initial distance
    dist_init = math.hypot(x_target-x_agent_init, y_target-y_agent_init)

    # Set the distance threshold 
    if time == 0 or dist_prev>dist_init or dist_prev == 0:
        epsilon = dist_init
    else:
        epsilon = dist_prev

    # Obtain the information about the agente
    for agent_id in range(obj.env.n_agents):
        agent = obj.env.get_unit_by_id(agent_id)
        agent_task = check_unit_task(obj,agent)
        if agent_task:
            x_agent = agent.pos.x
            y_agent = agent.pos.y
            dist = math.hypot(x_target-x_agent, y_target-y_agent)
            # Check if the agent get to the objetive
            if dist < obj.args.eps_objetive:
                reward_objetive = obj.args.reward_reach*obj.env.episode_limit - obj.args.obj_penal*task_reward
            elif dist < epsilon:
                dist_prev = dist
                reward_objetive = obj.args.goal_gamma*(1-dist/dist_init)
            elif dist > epsilon: 
                reward_objetive = 0
                
    max_reward = obj.args.reward_reach*obj.env.episode_limit
    reward_objetive /= max_reward / obj.env.reward_scale_rate

    return reward_objetive, dist, dist_prev

def reward_task_kill(obj):

    #Initialization
    reward_target = 0
    number_targets = 0
    deaths = 0
    kill = 0
    delta_enemy = 0
    delta_deaths = 0
    number_kill = 0
    # Target enemy damage and killing
    for e_id, e_unit in obj.env.enemies.items():
        agent_task = check_unit_task(obj,e_unit)
        if agent_task:
            number_targets += 1
            if not obj.env.death_tracker_enemy[e_id]:
                prev_health = (
                    obj.env.previous_enemy_units[e_id].health
                    + obj.env.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    obj.env.death_tracker_enemy[e_id] = 1
                    delta_deaths += (obj.env.reward_death_value*obj.args.reward_kill_target)
                    delta_enemy += (prev_health*obj.args.reward_hit_target)
                    deaths += 1
                    number_kill = np.count_nonzero(obj.env.death_tracker_enemy)
                else:
                    delta_enemy += (prev_health - e_unit.health - e_unit.shield) * obj.args.reward_hit_target

    reward_target += abs(delta_enemy + delta_deaths)
    
    # Set the max reward 
    max_reward = (
        (obj.env.n_enemies - number_targets) * obj.env.reward_death_value +
        (number_targets * obj.args.reward_kill_target * obj.env.reward_death_value) +
         obj.env.reward_win
        )

    reward_target /= max_reward / obj.env.reward_scale_rate

    if deaths == number_targets:
        kill = 1

    return reward_target, kill, number_kill


def reward_task_survive(obj,terminated,number_death):

    #Initialization
    reward_survive = 0
    survive = 0
    # Check if the agent has survived

    number_death = check_ally_death(obj,number_death)

    if number_death != 0 and terminated:
        reward_survive -= obj.args.death_penal*(obj.env.n_agents-number_death)*(obj.args.reward_task_survive/obj.env.n_agents)
    elif number_death == 0 and terminated:
        survive = 1
        reward_survive += obj.args.reward_task_survive
        # Check enemies that has survided
        dicc_state = obj.env.get_state_dict()
        enemies = dicc_state['enemies']
        n_enemies_alived = sum(enemies[:,0])
        reward_survive -= obj.args.penal_survive*n_enemies_alived*(obj.args.reward_task_survive/obj.env.n_enemies)
    elif number_death == 0 and not terminated:
        reward_survive = 0

    max_reward = obj.args.reward_task_survive

    reward_survive /= max_reward / obj.env.reward_scale_rate

    return reward_survive, survive, number_death

def reward_scalarization(obj,reward1,reward2):
    
    if obj.args.scalarization_method == 'lineal':
        reward1 = obj.args.weight1*reward1
        reward2 = obj.args.weight2*reward2
        reward = reward1 + reward2
    elif obj.args.scalarization_method == 'potencial':
        reward1 = reward1**obj.args.weight1
        reward2 = reward2**obj.args.weight2

    smac_reward = reward1
    task_reward = reward2

    return reward, smac_reward, task_reward

def check_unit_task(obj,unit):

    if obj.args.unit_task == 'Marine' and round(unit.health_max,1) == 45.0:
        agent_task = True
    elif obj.args.unit_task == 'Ghost' and round(unit.health_max,1) == 100.0:
        agent_task = True
    elif obj.args.unit_task == 'Marauder' and round(unit.health_max,1) == 125.0:
        agent_task = True
    elif obj.args.unit_task == 'SCV' and round(unit.health_max,1) == 45.1:
        agent_task = True
    else:
        agent_task = False
    
    return agent_task


def check_ally_death(obj,number_death):
    if number_death == 0:
        for agent_id in range(obj.env.n_agents):
            agent = obj.env.get_unit_by_id(agent_id)
            agent_task = check_unit_task(obj,agent)
            if agent_task:
                if agent.health == 0:
                    number_death = np.count_nonzero(obj.env.death_tracker_ally)
                else:
                    number_death = 0
    return number_death

def reward_task_survive_old(obj):

    #Initialization
    reward_survive = 0
    number_targets = 0
    survives = 0
    # Check if the agent has survived
    for agent_id in range(obj.env.n_agents):
        agent = obj.env.get_unit_by_id(agent_id)
        agent_task = check_unit_task(obj,agent)
        if agent_task:
            number_targets += 1
            if agent.health / agent.health_max > 0:
                #is alived
                survives += 1
                reward_survive += obj.args.reward_task_survive

    max_reward = obj.args.reward_task_survive

    reward_survive /= max_reward / obj.env.reward_scale_rate

    return reward_survive, number_targets, survives 

def reward_task_objetive_old(obj,terminated_task):

    # Initialization
    reward_objetive = 0

    x_agent_init = obj.env.x_init
    y_agent_init = obj.env.y_init
    # Get target point
    pos = obj.env.get_target_point(0)
    x_target = pos.x
    y_target = pos.y

    # Get initial distance
    dist_init = math.hypot(x_target-x_agent_init, y_target-y_agent_init)

    # Obtain the information about the agente
    for agent_id in range(obj.env.n_agents):
        agent = obj.env.get_unit_by_id(agent_id)
        if agent.unit_type == obj.args.agent_task:
            x_agent = agent.pos.x
            y_agent = agent.pos.y
            dist = math.hypot(x_target-x_agent, y_target-y_agent)

            # Check if the agent get to the objetive
            if dist < obj.args.eps_objetive:
                reward_objetive = obj.args.reward_reach*obj.env.episode_limit
                terminated_task = True
            else:
                reward_objetive = obj.args.goal_gamma*(1-dist/dist_init)

    max_reward = obj.args.reward_reach*obj.env.episode_limit
    reward_objetive /= max_reward / obj.env.reward_scale_rate

    return reward_objetive, terminated_task