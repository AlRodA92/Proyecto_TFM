import math

def reward_task_objetive(obj,terminated_task):

    # Initialization
    reward_objetive = 0

    x_agent_init = 15.5
    y_agent_init =  3.0

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

def reward_task_kill(obj):

    #Initialization
    reward_target = 0
    number_targets = 0
    delta_enemy = 0
    delta_deaths = 0
    # Target enemy damage and killing
    for e_id, e_unit in obj.env.enemies.items():
        if e_unit.unit_type == obj.args.enemy_target:
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

    return reward_target


def reward_task_survive(obj):

    #Initialization
    reward_survive = 0

    # Check if the agent has survived
    for agent_id in range(obj.env.n_agents):
        agent = obj.env.get_unit_by_id(agent_id)
        if agent.unit_type == obj.args.agent_survive:
            if agent.health / agent.health_max > 0:
                #is alived
                reward_survive += obj.args.reward_task_survive
                # Check enemies that has survided
                dicc_state = obj.env.get_state_dict()
                enemies = dicc_state['enemies']
                n_enemies_alived = sum(enemies[:,0])
                reward_survive -= n_enemies_alived*(obj.args.reward_task_survive/obj.env.n_enemies)

    max_reward = obj.args.reward_task_survive

    reward_survive /= max_reward / obj.env.reward_scale_rate

    return reward_survive    

def reward_scalarization(obj,reward1,reward2):

    if obj.args.scalarization_method == 'lineal':
        reward1 = obj.args.weight1*reward1
        reward2 = obj.args.weight2*reward2
        reward = reward1 + reward2
    elif obj.args.scalarization_method == 'potencial':
        reward1 = reward1**obj.args.weight1
        reward2 = reward2**obj.args.weight2
        reward = reward1 + reward2
    
    return reward, reward1, reward2