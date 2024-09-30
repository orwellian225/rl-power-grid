import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

weekdays_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def print_observation(observation):
    """
        print out a single observation / state in a neatly formatted manner
    """

    np.set_printoptions(linewidth=160, precision=2)

    date_str = f"{weekdays_map[observation['day_of_week']]} {observation['day']:02}/{observation['month']:02}/{observation['year']:04}"
    time_str = f"{observation['hour_of_day']:02}:{observation['minute_of_hour']:02}"
    datetime_str = f"{date_str} {time_str}"

    generator_production = str(observation['gen_p'])
    generator_voltage = str(observation['gen_v'])

    consumer_load = str(observation['load_p'])
    consumer_voltage = str(observation['load_v'])

    print(f"{'':=<180}")
    print(f"{datetime_str:^180}")
    print(f"{'':=<180}")

    print(f"{'Generator Information':^180}")
    print(f"{'':-<180}")
    print(f"Generator Production:{generator_production:>96} MW")
    print(f"Generator Bus Voltage:{generator_voltage:>95} kV")
    print(f"{'':-<180}")

    print(f"{'Consumer Information':^180}")
    print(f"{'':-<180}")
    print(f"Consumer Load:{consumer_load:>103} MW")
    print(f"Consumer Bus Voltage:{consumer_voltage:>95} kV")
    print(f"{'':-<180}")

    print(f"{'Powerline Information':^180}")
    print(f"{'':-<180}")
    print(f"Mean Origin Active Power Flow:{np.mean(observation['p_or'])} MW")
    print(f"{'':-<180}")
    print(f"{'':=<180}")

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