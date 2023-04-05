import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib.animation import FuncAnimation
import random

################################# Constants ##############################
INTERACTION_STRENGTH = 1
MAGNETIC_FIELD = 0
TEMPERATURE = 0.8
SPINS_IN_ROW = 100
SPINS = SPINS_IN_ROW**2
FINAL_STEP = 500000
########################################################################


def spin_system(spins_in_row):
    """
    Create a two-dimensional array representing a 2D spin system, with each element assigned a random value of 1 or -1.

    Parameters:
    spins_in_row: int - number of spins in a single row or column of the 2D array.

    Returns:
    system: numpy.ndarray - a 2D array with dimensions (spins_in_row, spins_in_row), where each element has a value of either 1 or -1.
    """
    system = 2 * np.random.randint(2, size=(spins_in_row, spins_in_row)) - 1
    return system


class SpinEnergies:
    """
    A class for calculating energy of the spin system, as well as individual spin energies.
    """

    def __init__(self, interaction_strength, magnetic_field, spins_in_row) -> None:
        self.interaction_strength = interaction_strength
        self.magnetic_field = magnetic_field
        self.spins_in_row = spins_in_row

    def individual_energy(self, i, j, configuration):
        """
        Calculate the energy of an individual spin with four neighbours.

        Args:
            i (int): index 1 of the spin
            j (int): index 2 of the spin
            configuration (ndarray): the spin configuration

        Returns:
            energy (float): the energy of the individual spin
        """
        neighbour_spins = (
            configuration[(i + 1) % self.spins_in_row, j]
            + configuration[i, (j + 1) % self.spins_in_row]
            + configuration[(i - 1) % self.spins_in_row, j]
            + configuration[i, (j - 1) % self.spins_in_row]
        )
        energy = -self.interaction_strength * configuration[i, j] * neighbour_spins
        return energy

    def system_energy(self, configuration):
        """
        Calculate the total interaction energy of the spin configuration with periodic boundary condition.

        Args:
            configuration (ndarray): the spin configuration

        Returns:
            energy (float): the total energy of the configuration
        """
        neighbour_energy = (
            np.roll(configuration, 1, axis=0)
            + np.roll(configuration, -1, axis=0)
            + np.roll(configuration, 1, axis=1)
            + np.roll(configuration, -1, axis=1)
        )
        energy = np.sum(
            -self.interaction_strength * configuration * neighbour_energy
        ) / 2 - self.magnetic_field * np.sum(configuration)
        return energy


class MonteCarlo:
    """
    This class performs Metropolis calculations.
    """

    def __init__(
        self, interaction_strength, magnetic_field, spins_in_row, temperature
    ) -> None:
        """
        Initializes the MonteCarlo object.

        Parameters:
        -----------
        interaction_strength : float
            The strength of the interaction between neighboring spins.
        magnetic_field : float
            The strength of the external magnetic field.
        spins_in_row : int
            The number of spins in each row of the spin configuration.
        temperature : float
            The temperature at which the calculations are performed.
        """
        self.interaction_strength = interaction_strength
        self.magnetic_field = magnetic_field
        self.spins_in_row = spins_in_row
        self.temperature = temperature

    def spin_flip(self, configuration):
        """
        Flips a random spin in the spin configuration.

        Parameters:
        -----------
        configuration : ndarray
            The current spin configuration.

        Returns:
        --------
        old_energy : float
            The individual energy of the flipped spin before flipping.
        new_energy : float
            The individual energy of the flipped spin after flipping.
        new_configuration : ndarray
            The new spin configuration with the flipped spin.
        """
        i = np.random.randint(self.spins_in_row)
        j = np.random.randint(self.spins_in_row)
        new_configuration = configuration.copy()
        new_configuration[i, j] = -1 * new_configuration[i, j]
        old_energy = SpinEnergies(
            self.interaction_strength, self.magnetic_field, self.spins_in_row
        ).individual_energy(i, j, configuration)
        new_energy = (
            -1 * old_energy
        )  # SpinEnergies(self.interaction_strength, self.magnetic_field, self.spins_in_row).individual_energy(i, j, new_configuration)
        return old_energy, new_energy, new_configuration

    def accept_reject(self, configuration):
        """
        Controls the acceptance and rejection of spin flips.

        Parameters:
        -----------
        configuration : ndarray
            The current spin configuration.

        Returns:
        --------
        accepted : bool
            True if the flip is accepted, False otherwise.
        old_energy : float
            The individual energy of the flipped spin before flipping.
        new_energy : float
            The individual energy of the flipped spin after flipping.
        new_configuration : ndarray
            The new spin configuration with the flipped spin.
        """
        old_energy, new_energy, new_configuration = self.spin_flip(configuration)
        if random.random() < np.exp((old_energy - new_energy) / self.temperature):
            return True, old_energy, new_energy, new_configuration
        else:
            return False, old_energy, new_energy, new_configuration


count = 0

INITIAL_CONFIGURATIONS = spin_system(SPINS_IN_ROW)
INITIAL_TOTAL_ENERGY = SpinEnergies(
    INTERACTION_STRENGTH, MAGNETIC_FIELD, SPINS_IN_ROW
).system_energy(INITIAL_CONFIGURATIONS)

total_energy = INITIAL_TOTAL_ENERGY.copy()
total_energy_array = [total_energy]
configurations = INITIAL_CONFIGURATIONS.copy()
all_configurations = [configurations]

# plt.figure()
# plt.pcolor(INITIAL_CONFIGURATIONS, cmap="winter")
# plt.colorbar()

while count < FINAL_STEP:
    initiation = MonteCarlo(
        INTERACTION_STRENGTH, MAGNETIC_FIELD, SPINS_IN_ROW, TEMPERATURE
    ).accept_reject(configurations)
    if initiation[0] == True:
        configurations = initiation[3]
        total_energy = total_energy + initiation[2] - initiation[1]
        total_energy_array.append(total_energy)
        all_configurations.append(configurations)
        # print(count)
    count += 1


def update(frame):
    configuration = all_configurations[frame * 100]
    ax.clear()
    contour = ax.pcolor(configuration, cmap="winter")
    return contour


fig, ax = plt.subplots()

animation = FuncAnimation(
    fig, update, frames=int(len(all_configurations) / 100), interval=0.0002
)

# Show the animation
plt.show()

# plt.figure()
# plt.pcolor(configurations, cmap="winter")
# plt.colorbar()

# plt.figure()
# plt.plot(total_energy_array)
# plt.show()
