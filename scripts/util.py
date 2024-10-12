import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

weekdays_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def get_formatted_date(observation):
    date_str = f"{weekdays_map[observation['day_of_week']]} {observation['day']:02}/{observation['month']:02}/{observation['year']:04}"
    time_str = f"{observation['hour_of_day']:02}:{observation['minute_of_hour']:02}"
    datetime_str = f"{date_str} {time_str}"

    return datetime_str

def print_observation(observation):
    """
        print out a single observation / state in a neatly formatted manner
    """

    np.set_printoptions(linewidth=160, precision=2)


    print(f"{'':=<120}")
    print(f"{get_formatted_date(observation):^120}")

    print(f"{'':-<120}")
    print(f"{'Grid Information':^120}")
    print(f"{'':-<120}")
    print(f"Grid Topology: {observation['topo_vect']}")
    print(f"Grid Attention Budget: {observation['attention_budget']}")
    print(f"{'':-<120}")

    print(f"{'':-<120}")
    print(f"{'Generator Information':^120}")
    print(f"{'':-<120}")
    print(f"Number of Generators: {observation['gen_p'].shape[0]}")
    print(f"Generator Active Production: {str(observation['gen_p'])} MW")
    print(f"Generator Reactive Production: {str(observation['gen_q'])} MVar")
    print(f"Generator Bus Voltage: {str(observation['gen_v'])} kV")
    print(f"Generator Voltage Angle: {str(observation['gen_theta'])} degrees")
    print(f"Generator Active Production before Curtailment: {str(observation['gen_p_before_curtail'])} MW")
    print(f"Generator Bus Voltage: {str(observation['gen_v'])} kV")
    print(f"Generator Margin Up: {str(observation['gen_margin_up'])}")
    print(f"Generator Margin Down: {str(observation['gen_margin_down'])}")
    print(f"Generator Target Dispatch: {str(observation['target_dispatch'])}")
    print(f"Generator Actual Dispatch: {str(observation['actual_dispatch'])}")
    print(f"{'':-<120}")

    print(f"{'Consumer Information':^120}")
    print(f"{'':-<120}")
    print(f"Number of Consumers: {observation['load_p'].shape[0]}")
    print(f"Consumer Active Consumption: {observation['load_p']} MW")
    print(f"Consumer Reactive Consumption: {observation['load_q']} MVar")
    print(f"Consumer Bus Voltage: {observation['load_v']} kV")
    print(f"Consumer Voltage Angle: {str(observation['load_theta'])} degrees")
    print(f"{'':-<120}")


    print(f"{'Powerline Information':^120}")
    print(f"{'':-<120}")
    print(f"Number of powerlines: {observation['rho'].shape[0]}")
    print(f"Time till next maintainence: {observation['time_next_maintenance']}")
    print(f"Duration of next maintainence: {observation['duration_next_maintenance']}")
    print(f"Capacity: {observation['rho']}")
    print(f"Thermal Limit: {observation['thermal_limit']}")
    print(f"Status: {observation['line_status'].astype(np.uint8)}")
    print(f"Time since overflow: {observation['timestep_overflow']}")
    print(f"{'':-<120}")
    print(f"Origin Active Power Flow: {observation['p_or']} MW")
    print(f"Origin Reactive Power Flow: {observation['q_or']} MVar")
    print(f"Origin Voltage: {observation['v_or']} kV")
    print(f"Origin Voltage Angle: {observation['theta_or']} degrees")
    print(f"Origin Current: {observation['a_or']} A")
    print(f"{'':-<120}")
    print(f"Extremity Active Power Flow: {observation['p_ex']} MW")
    print(f"Extremity Reactive Power Flow: {observation['q_ex']} MVar")
    print(f"Extremity Voltage: {observation['v_ex']} kV")
    print(f"Extremity Voltage Angle: {observation['theta_ex']} degrees")
    print(f"Extremity Current: {observation['a_ex']} A")
    print(f"{'':-<120}")

    print(f"{'':=<120}")

def visualize_powergrid(obs, ax=None):
    """
        Draw the powergrid network on the provided matplotlib axes
    """
    date_str = f"{weekdays_map[obs['day_of_week']]} {obs['day']:02}/{obs['month']:02}/{obs['year']:04}"
    time_str = f"{obs['hour_of_day']:02}:{obs['minute_of_hour']:02}"
    datetime_str = f"{date_str} {time_str}"

    if ax is None:
        ax = plt.subplot('1')

    ax.set_aspect('equal')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(datetime_str)

    class Node:
        def __init__(self, active_power: float, reactive_power: float, voltage_mag: float, voltage_angle: float, substation_id: int, cooldown: int):
            self.active_power = active_power
            self.reactive_power = reactive_power
            self.voltage_mag = voltage_mag
            self.voltage_angle = voltage_angle
            self.substation_id = substation_id
            self.cooldown = cooldown
        
        def draw_at(self, ax, x: float, y: float):
            active_color = 'orange' if self.active_power > 0 else 'b'

            active = mpatches.Circle((x,y), radius=2., color=active_color)

            ax.add_patch(active)

    return ax