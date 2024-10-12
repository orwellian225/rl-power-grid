import numpy as np

def gym_to_model_observation(obs):
    feature_vec = np.concatenate([
        obs['gen_p'], obs['gen_q'], obs['gen_v'], obs['gen_theta'], obs['gen_p_before_curtail'],
        obs['gen_margin_up'], obs['gen_margin_down'], 
        obs['target_dispatch'], obs['actual_dispatch'],
        obs['load_p'], obs['load_q'], obs['load_v'], obs['load_theta'],
        obs['rho'], 
        # obs['time_next_maintenance'], obs['duration_next_maintenance'], # DISABLED no maintenance in current environment 
        obs['thermal_limit'], obs['line_status'].astype(np.uint8), obs['timestep_overflow'],
        obs['p_or'], obs['q_or'], obs['v_or'], obs['theta_or'], obs['a_or'],
        obs['p_ex'], obs['q_ex'], obs['v_ex'], obs['theta_ex'], obs['a_ex'],
    ])

    return feature_vec

def gym_to_model_action(action, num_bus_objs, num_lines):
    action_vec = np.concatenate([
        np.zeros(num_bus_objs), # Modifying Bus Objects
        np.zeros(num_lines) # Modifying line status
    ])

    for i, a_val in enumerate(action['set_bus']):
        action_vec[i] = a_val

    for i, a_val in enumerate(action['set_line_status']):
        action_vec[num_bus_objs + i] = a_val

    return action_vec

def model_to_gym_action(action_vec, num_bus_objs, num_generators, num_lines):
    return {
        "change_bus": np.zeros(num_bus_objs).astype(np.bool_),
        "change_line_status": np.zeros(num_lines).astype(np.bool_),
        "curtail": np.zeros(num_generators).astype(np.float32),
        "redispatch": np.zeros(num_generators).astype(np.float32),
        "set_bus": action_vec[:num_bus_objs],
        "set_line_status": action_vec[num_bus_objs:]
    }
