import numpy as np
import matplotlib.pyplot as plt
import random

############### Constants###################
INTERACTION_STRENGTH = 1
INTERACTION_RANGE = 5
ALPHA = 2
DISTANCE = 1.0
MAGNETIC_FIELD = 0
TEMPERATURE = 0.8
SPINS_IN_ROW = 100
SPINS = SPINS_IN_ROW**2
FINAL_STEP = 100000
###########################################


def spin_system(num_spins_row):
    """
    Generate a spin system on a two-dimensional crystal.

    Args:
    - num_spins_row (int): The number of spins in one row/dimension.

    Returns:
    - configuration (np.ndarray): A 2D array with dimensions num_spins_row x num_spins_row where each element
    takes a value of +1 or -1.
    """
    configuration = 2 * np.random.randint(2, size=(num_spins_row, num_spins_row)) - 1
    return configuration


class EnergyOfSystem:
    """
    Calculates the energy of a spin system based on its configuration and physical parameters.
    """

    def __init__(self, num_spins_row, distance, alpha, int_strength, int_range):
        """
        Initialize EnergyOfSystem class.

        Parameters:
        -----------
        num_spins_row : int
            The number of spins in one row/dimension of the 2D array representing the spin system.
        distance : float
            The distance between neighboring spins.
        alpha : float
            A parameter controlling the range of interaction between spins.
        int_strength : float
            The strength of the interaction between neighboring spins.
        int_range : int
            The range of interaction between neighboring spins.
        """
        self.num_spins_row = num_spins_row
        self.distance = distance
        self.alpha = alpha
        self.int_strength = int_strength
        self.int_range = int_range

    def spin_energy(self, configuration, spin_index1, spin_index2):
        """
        Calculates the energy of a spin based on its interaction with neighboring spins.

        Parameters:
        -----------
        configuration : numpy.ndarray
            A 2D array with dimensions (num_spins_row, num_spins_row) representing the spin system configuration.
        spin_index1 : int
            The row index of the spin whose energy is to be calculated.
        spin_index2 : int
            The column index of the spin whose energy is to be calculated.

        Returns:
        --------
        s_energy : float
            The energy of the spin.
        """
        s_energy = 0
        for i in range(spin_index1 - self.int_range, spin_index1 + self.int_range + 1):
            for j in range(
                spin_index2 - self.int_range, spin_index2 + self.int_range + 1
            ):
                if i != spin_index1 or j != spin_index2:
                    d_index1 = (
                        np.abs(i % self.num_spins_row - spin_index1)
                        - np.round(
                            np.abs(i % self.num_spins_row - spin_index1)
                            / self.num_spins_row
                        )
                        * self.num_spins_row
                    )  #% self.num_spins_row
                    d_index2 = (
                        np.abs(j % self.num_spins_row - spin_index2)
                        - np.round(
                            np.abs(j % self.num_spins_row - spin_index2)
                            / self.num_spins_row
                        )
                        * self.num_spins_row
                    )  #% self.num_spins_row
                    s_energy += (
                        configuration[i % self.num_spins_row, j % self.num_spins_row]
                        * self.int_strength
                        / (
                            np.sqrt(
                                self.distance**2 * d_index1**2
                                + self.distance**2 * d_index2**2
                            )
                            ** self.alpha
                        )
                    )
        return -s_energy * configuration[spin_index1, spin_index2]

    def total_energy(self, configuration):
        """
        Calculates the total energy of a spin system based on its configuration and physical parameters.

        Parameters:
        -----------
        configuration : numpy.ndarray
            A 2D array with dimensions (num_spins_row, num_spins_row) representing the spin system configuration.

        Returns:
        --------
        t_energy : float
            The total energy of the spin system.
        """
        t_energy = 0
        for i in range(self.num_spins_row):
            for j in range(self.num_spins_row):
                t_energy += self.spin_energy(configuration, i, j)
        return t_energy


class MonteCarlo:
    """
    This class implements the Monte Carlo simulation algorithm for a spin system.
    --------------------------------------
    Inputs:
    int_strength: Interaction strength parameter.
    distance: Distance between spins.
    alpha: Exponent in the interaction strength formula.
    num_spins_row: The number of spins in one row/dimension.
    temperature: Temperature of the system.
    int_range: Interaction range parameter.
    """

    def __init__(
        self, int_strength, distance, alpha, num_spins_row, temperature, int_range
    ):
        self.int_strength = int_strength
        self.num_spins_row = num_spins_row
        self.temperature = temperature
        self.distance = distance
        self.alpha = alpha
        self.int_range = int_range

    def pick_n_flip(self, configuration):
        """
        This function picks a random spin, flips it and returns the new configuration.
        --------------------------------------
        Inputs:
        configuration: A 2D array with dimension N x N where each element takes a value +1 or -1.
        --------------------------------------
        Outputs:
        old_energy: Energy of the spin before it was flipped.
        new_energy: Energy of the spin after it was flipped.
        new_configuration: A new configuration after the spin was flipped.
        """
        s_index1, s_index2 = np.random.randint(self.num_spins_row), np.random.randint(
            self.num_spins_row
        )
        new_configuration = configuration.copy()
        new_configuration[s_index1, s_index2] = (
            -1 * configuration[s_index1, s_index2].copy()
        )
        old_energy = EnergyOfSystem(
            self.num_spins_row,
            self.distance,
            self.alpha,
            self.int_strength,
            self.int_range,
        ).spin_energy(configuration, s_index1, s_index2)
        new_energy = (
            -1 * old_energy
        )  # EnergyOfSystem(self.num_spins_row, self.distance, self.alpha, self.int_strength).spin_energy(new_configuration, s_index1, s_index2)
        return old_energy, new_energy, new_configuration

    def flip_decision(self, configuration, total_energy):
        """
        This function decides whether to flip the randomly selected spin based on energy conditions.
        --------------------------------------
        Inputs:
        configuration: A 2D array with dimension N x N where each element takes a value +1 or -1.
        total_energy: Current energy of the system.
        --------------------------------------
        Outputs:
        True or False: Whether to flip the selected spin.
        new_configuration: A new configuration after the spin was flipped.
        total_energy: Updated energy of the system.
        """
        old_energy, new_energy, new_configuration = self.pick_n_flip(configuration)
        # print(old_energy)
        random_decider = random.random()
        if random_decider < np.exp((old_energy - new_energy) / self.temperature):
            # print(
            #     random_decider,
            #     np.exp((old_energy - new_energy) / self.temperature),
            #     old_energy,
            # )
            return True, new_configuration, total_energy + (new_energy - old_energy)
        else:
            return False, configuration, total_energy


count = 0
INITIAL_CONFIGURATIONS = spin_system(SPINS_IN_ROW)
INITIAL_TOTAL_ENERGY = EnergyOfSystem(
    SPINS_IN_ROW, DISTANCE, ALPHA, INTERACTION_STRENGTH, INTERACTION_RANGE
).total_energy(INITIAL_CONFIGURATIONS)

total_energy = INITIAL_TOTAL_ENERGY.copy()
# print(
#     EnergyOfSystem(SPINS_IN_ROW, DISTANCE, ALPHA, INTERACTION_STRENGTH).total_energy(
#         np.ones([SPINS_IN_ROW, SPINS_IN_ROW])
#     )
# )
total_energy_array = [total_energy]
configurations = INITIAL_CONFIGURATIONS.copy()

plt.figure()
plt.pcolor(INITIAL_CONFIGURATIONS, cmap="winter")
plt.colorbar()
# plt.show()

while count < FINAL_STEP:
    initiation = MonteCarlo(
        INTERACTION_STRENGTH,
        DISTANCE,
        ALPHA,
        SPINS_IN_ROW,
        TEMPERATURE,
        INTERACTION_RANGE,
    ).flip_decision(configurations, total_energy)
    if initiation[0] == True:
        configurations = initiation[1]
        total_energy = initiation[2]
        total_energy_array.append(total_energy)
        print(count)
    count += 1

plt.figure()
plt.pcolor(configurations, cmap="winter")
plt.colorbar()

plt.figure()
plt.plot(total_energy_array)
plt.show()
