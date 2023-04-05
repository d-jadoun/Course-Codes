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
SPINS_IN_ROW = 50
SPINS = SPINS_IN_ROW**2
FINAL_STEP = 250000
MAXIMUM_ROTATION = np.pi / 5
###########################################


def spin_system(spins_in_row):
    """
    A spin system with each spin having xy orientation (cos(theta),sin(theta))
    is constructed here.
    -------------------------------------------------------------------------
    Inputs
    spins_in_row: the number of spins in each row.
    -------------------------------------------------------------------------
    returns
    system: a 2D array with each element having random xy orientation.
    """
    system = np.random.uniform(0, 2 * np.pi, [spins_in_row, spins_in_row])
    return system


class EnergyOfSystem:
    """
    A class to calculate the energy of a given configuration of spins in a spin system.
    """

    def __init__(self, num_spins_row, distance, alpha, int_strength, int_range):
        """
        Initialize the EnergyOfSystem class instance.

        Parameters:
        -----------
        num_spins_row : int
            The number of spins in each row of the 2D spin system.
        distance : float
            The distance between the spins in the spin system.
        alpha : float
            The decay rate of the interaction strength with distance.
        int_strength : float
            The interaction strength between the spins in the spin system.
        int_range : int
            The range of interaction between the spins in the spin system.
        """
        self.num_spins_row = num_spins_row
        self.distance = distance
        self.alpha = alpha
        self.int_strength = int_strength
        self.int_range = int_range

    def spin_energy(self, configuration, spin_index1, spin_index2):
        """
        Calculate the energy of a given spin in the spin system.

        Parameters:
        -----------
        configuration : numpy.ndarray
            A 2D numpy array representing the spin configuration of the spin system.
        spin_index1 : int
            The row index of the spin in the spin system.
        spin_index2 : int
            The column index of the spin in the spin system.

        Returns:
        --------
        s_energy : float
            The energy of the given spin in the spin system.
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
                        (
                            np.cos(configuration[spin_index1, spin_index2])
                            * np.cos(
                                configuration[
                                    i % self.num_spins_row, j % self.num_spins_row
                                ]
                            )
                            + np.sin(configuration[spin_index1, spin_index2])
                            * np.sin(
                                configuration[
                                    i % self.num_spins_row, j % self.num_spins_row
                                ]
                            )
                        )
                        * self.int_strength
                        / (
                            np.sqrt(
                                self.distance**2 * d_index1**2
                                + self.distance**2 * d_index2**2
                            )
                            ** self.alpha
                        )
                    )
        return -s_energy

    def total_energy(self, configuration):
        """
        Calculate the total energy of the spin system configuration.

        Parameters:
        -----------
        configuration : numpy.ndarray
            A 2D numpy array representing the spin configuration of the spin system.

        Returns:
        --------
        t_energy : float
            The total energy of the given spin system configuration.
        """
        t_energy = 0
        for i in range(self.num_spins_row):
            for j in range(self.num_spins_row):
                t_energy += self.spin_energy(configuration, i, j)
        return t_energy


class MonteCarlo:
    """
    Class for implementing the Monte Carlo simulation.
    """

    def __init__(
        self,
        int_strength,
        distance,
        alpha,
        num_spins_row,
        temperature,
        int_range,
        max_rotation,
    ) -> None:
        """
        Initializes the Monte Carlo simulation with the given parameters.

        Args:
        - int_strength: float, strength of interaction between spins
        - distance: float, distance between spins
        - alpha: float, distance scaling factor for interaction strength
        - num_spins_row: int, number of spins in each row
        - temperature: float, temperature of the system
        - int_range: int, range of interaction between spins
        - max_rotation: float, maximum amount by which a spin can be rotated randomly
        """
        self.int_strength = int_strength
        self.num_spins_row = num_spins_row
        self.temperature = temperature
        self.distance = distance
        self.alpha = alpha
        self.int_range = int_range
        self.max_rotation = max_rotation

    def spin_rotate(self, configuration):
        """
        Rotates a randomly selected spin in the given configuration by a random angle
        and returns the old and new energies of the spin and the new configuration.

        Args:
        - configuration: ndarray, 2D array of spins

        Returns:
        - old_energy: float, energy of the spin before rotation
        - new_energy: float, energy of the spin after rotation
        - new_configuration: ndarray, 2D array of spins after rotation
        """
        i = np.random.randint(self.num_spins_row)
        j = np.random.randint(self.num_spins_row)
        new_configuration = configuration.copy()
        old_orientation = configuration[i, j].copy()
        new_orientation = 0 * old_orientation
        rotation = self.max_rotation * (random.random() - 0.5)
        new_orientation = old_orientation + rotation
        new_configuration[i, j] = new_orientation
        new_configuration = new_configuration % (2 * np.pi)
        old_energy = EnergyOfSystem(
            self.num_spins_row,
            self.distance,
            self.alpha,
            self.int_strength,
            self.int_range,
        ).spin_energy(configuration, i, j)
        new_energy = EnergyOfSystem(
            self.num_spins_row,
            self.distance,
            self.alpha,
            self.int_strength,
            self.int_range,
        ).spin_energy(new_configuration, i, j)
        return old_energy, new_energy, new_configuration

    def flip_decision(self, configuration, total_energy):
        """
        Decides whether to accept or reject a proposed spin rotation based on the
        energy difference between the old and new configurations and the temperature
        of the system.

        Args:
        - configuration: ndarray, 2D array of spins
        - total_energy: float, total energy of the system

        Returns:
        - bool, True if the proposed spin rotation is accepted, False otherwise
        - configuration: ndarray, 2D array of spins after the proposed spin rotation
        - total_energy: float, total energy of the system after the proposed spin rotation
        """
        old_energy, new_energy, new_configuration = self.spin_rotate(configuration)
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
# print(total_energy)
# print(
#     EnergyOfSystem(SPINS_IN_ROW, DISTANCE, ALPHA, INTERACTION_STRENGTH).total_energy(
#         np.ones([SPINS_IN_ROW, SPINS_IN_ROW])
#     )
# )
total_energy_array = [total_energy]
configurations = INITIAL_CONFIGURATIONS.copy()

x_axis = np.arange(0, SPINS_IN_ROW, 1)
y_axis = np.arange(0, SPINS_IN_ROW, 1)
X_mesh, Y_mesh = np.meshgrid(x_axis, y_axis, indexing="ij")

plt.figure()
plt.quiver(
    X_mesh,
    Y_mesh,
    np.cos(INITIAL_CONFIGURATIONS),
    np.sin(INITIAL_CONFIGURATIONS),
    INITIAL_CONFIGURATIONS,
    headaxislength=3,
    headwidth=4,
    scale=50,
    cmap="hsv",
    width=0.0015,
)
plt.colorbar()


while count < FINAL_STEP:
    initiation = MonteCarlo(
        INTERACTION_STRENGTH,
        DISTANCE,
        ALPHA,
        SPINS_IN_ROW,
        TEMPERATURE,
        INTERACTION_RANGE,
        MAXIMUM_ROTATION,
    ).flip_decision(configurations, total_energy)
    if initiation[0] == True:
        configurations = initiation[1]
        total_energy = initiation[2]
        total_energy_array.append(total_energy)
        print(count)
    count += 1


plt.figure()
plt.quiver(
    X_mesh,
    Y_mesh,
    np.cos(configurations),
    np.sin(configurations),
    configurations,
    headaxislength=3,
    headwidth=4,
    scale=50,
    cmap="hsv",
    width=0.0015,
)
plt.colorbar()

plt.figure()
plt.plot(total_energy_array)
plt.show()
