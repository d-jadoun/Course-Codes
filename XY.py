import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import random

################################# Constants ##############################
INTERACTION_STRENGTH = 1
MAGNETIC_FIELD = 0
TEMPERATURE = 0.4
SPINS_IN_ROW = 15
SPINS = SPINS_IN_ROW**2
FINAL_STEP = 1000000
MAXIMUM_ROTATION = np.pi / 5
########################################################################


def spin_system(spins_in_row):
    """
    Constructs a spin system with each spin oriented along the xy-plane, where
    the orientation of each spin is specified by the pair (cos(theta), sin(theta)).

    Parameters
    ----------
    spins_in_row : int
        The number of spins in each row of the system.

    Returns
    -------
    system : ndarray
        A 2D array with each element having a random (cos(theta), sin(theta))
        orientation.
    """
    system = np.random.uniform(0, 2 * np.pi, [spins_in_row, spins_in_row])
    return system


class SpinEnergy:
    """
    This class calculates the total energy of the system along with
    the energy of individual spins.
    """

    def __init__(
        self, interaction_strength, magnetic_field, spins_in_row, maximum_rotation
    ) -> None:
        """
        Initialize the spin system with the given interaction strength, magnetic field, spins in a row,
        and the maximum rotation allowed for each spin.
        """
        self.interaction_strength = interaction_strength
        self.magnetic_field = magnetic_field
        self.spins_in_row = spins_in_row
        self.maximum_rotation = maximum_rotation

    def individual_energy(self, i, j, configuration):
        """
        Calculate the energy of an individual spin in the system.

        Args:
        i,j: The indices of the spin
        configuration: The spin system configuration

        Returns:
        The energy of the individual spin.
        """
        neighbour_spins_cosine = (
            np.cos(configuration[(i + 1) % self.spins_in_row, j])
            + np.cos(configuration[i, (j + 1) % self.spins_in_row])
            + np.cos(configuration[(i - 1) % self.spins_in_row, j])
            + np.cos(configuration[i, (j - 1) % self.spins_in_row])
        )
        neighbour_spins_sine = (
            np.sin(configuration[(i + 1) % self.spins_in_row, j])
            + np.sin(configuration[i, (j + 1) % self.spins_in_row])
            + np.sin(configuration[(i - 1) % self.spins_in_row, j])
            + np.sin(configuration[i, (j - 1) % self.spins_in_row])
        )
        energy = -self.interaction_strength * (
            np.cos(configuration[i, j]) * neighbour_spins_cosine
            + np.sin(configuration[i, j]) * neighbour_spins_sine
        )
        return energy

    def system_energy(self, configuration):
        """
        Calculate the total energy of the spin system.

        Args:
        configuration: The spin system configuration.

        Returns:
        The total energy of the spin system.
        """
        neighbour_energy_cosine = (
            np.cos(np.roll(configuration, 1, axis=1))
            + np.cos(np.roll(configuration, -1, axis=1))
            + np.cos(np.roll(configuration, 1, axis=0))
            + np.cos(np.roll(configuration, -1, axis=0))
        )
        neighbour_energy_sine = (
            np.sin(np.roll(configuration, 1, axis=1))
            + np.sin(np.roll(configuration, -1, axis=1))
            + np.sin(np.roll(configuration, 1, axis=0))
            + np.sin(np.roll(configuration, -1, axis=0))
        )
        energy = (
            np.sum(
                -self.interaction_strength
                * (
                    np.cos(configuration) * neighbour_energy_cosine
                    + np.sin(configuration)
                    + neighbour_energy_sine
                )
            )
            / 2
        )
        return energy


class MonteCarlo:
    """
    This class is used to perform a Monte-Carlo run for the XY model.
    """

    def __init__(
        self,
        interaction_strength,
        magnetic_field,
        spins_in_row,
        temperature,
        maximum_rotation,
    ) -> None:
        """
        Initializes a SpinEnergy object with given parameters.

        Parameters:
        interaction_strength: float, the strength of interaction between neighboring spins.
        magnetic_field: float, the strength of the external magnetic field.
        spins_in_row: int, the number of spins in each row of the spin system.
        temperature: float, the temperature of the system.
        maximum_rotation: float, the maximum rotation allowed for a spin during Monte Carlo simulation.
        """
        self.interaction_strength = interaction_strength
        self.magnetic_field = magnetic_field
        self.spins_in_row = spins_in_row
        self.temperature = temperature
        self.maximum_rotation = maximum_rotation

    def spin_rotate(self, configuration):
        """
        Rotates a randomly selected spin and returns the old and new energies along with new spin configuration.

        Parameters:
        configuration: numpy array, the current spin configuration.

        Returns:
        old_energy: float, the energy of the system before rotation.
        new_energy: float, the energy of the system after rotation.
        new_configuration: numpy array, the new spin configuration after rotation.
        """
        i = np.random.randint(self.spins_in_row)
        j = np.random.randint(self.spins_in_row)
        new_configuration = configuration.copy()
        old_orientation = configuration[i, j].copy()
        new_orientation = 0 * old_orientation
        rotation = self.maximum_rotation * (random.random() - 0.5)
        new_orientation = old_orientation + rotation
        new_configuration[i, j] = new_orientation
        new_configuration = new_configuration % (2 * np.pi)
        old_energy = SpinEnergy(
            self.interaction_strength,
            self.magnetic_field,
            self.spins_in_row,
            self.maximum_rotation,
        ).individual_energy(i, j, configuration)
        new_energy = SpinEnergy(
            self.interaction_strength,
            self.magnetic_field,
            self.spins_in_row,
            self.maximum_rotation,
        ).individual_energy(i, j, new_configuration)
        return old_energy, new_energy, new_configuration

    def accept_reject(self, configuration):
        """
        Decides whether to accept or reject a proposed spin rotation.

        Parameters:
        configuration: numpy array, the current spin configuration.

        Returns:
        accepted: bool, whether the proposed spin rotation is accepted or rejected.
        old_energy: float, the energy of the system before rotation.
        new_energy: float, the energy of the system after rotation.
        new_configuration: numpy array, the new spin configuration after rotation.
        old_configuration: numpy array, the old spin configuration before rotation.
        """
        old_energy, new_energy, new_configuration = self.spin_rotate(configuration)
        if random.random() < np.exp((old_energy - new_energy) / self.temperature):
            return True, old_energy, new_energy, new_configuration
        else:
            return False, old_energy, new_energy, new_configuration


count = 0

INITIAL_CONFIGURATIONS = spin_system(SPINS_IN_ROW)
INITIAL_TOTAL_ENERGY = SpinEnergy(
    INTERACTION_STRENGTH, MAGNETIC_FIELD, SPINS_IN_ROW, MAXIMUM_ROTATION
).system_energy(INITIAL_CONFIGURATIONS)

total_energy = INITIAL_TOTAL_ENERGY.copy()
total_energy_array = [total_energy]
configurations = INITIAL_CONFIGURATIONS.copy()
all_configurations = [configurations]

x_axis = np.arange(0, SPINS_IN_ROW, 1)
y_axis = np.arange(0, SPINS_IN_ROW, 1)
X_mesh, Y_mesh = np.meshgrid(x_axis, y_axis, indexing="ij")

# plt.figure()
# plt.quiver(
#     X_mesh,
#     Y_mesh,
#     np.cos(INITIAL_CONFIGURATIONS),
#     np.sin(INITIAL_CONFIGURATIONS),
#     INITIAL_CONFIGURATIONS,
#     headaxislength=3,
#     headwidth=4,
#     scale=50,
#     cmap="hsv",
#     width=0.0015,
# )
# plt.colorbar()

# plt.figure()
# plt.imshow(INITIAL_CONFIGURATIONS, cmap="seismic")
# plt.colorbar()
# plt.axis("off")


while count < FINAL_STEP:
    initiation = MonteCarlo(
        INTERACTION_STRENGTH,
        MAGNETIC_FIELD,
        SPINS_IN_ROW,
        TEMPERATURE,
        MAXIMUM_ROTATION,
    ).accept_reject(configurations)
    if initiation[0] == True:
        configurations = initiation[3]
        total_energy = total_energy + initiation[2] - initiation[1]
        total_energy_array.append(total_energy)
        if count % 10 == 0:
            all_configurations.append(configurations)
        # print(count)
    count += 1


colors = cm.hsv(configurations / (2 * np.pi))


# plt.figure()
# plt.imshow(configurations, cmap="seismic")
# plt.colorbar()
# plt.axis("off")


# plt.figure()
# plt.plot(total_energy_array)
# plt.show()


def update(frame):
    configurations = all_configurations[frame * 10]
    ax.clear()
    contour = ax.quiver(
        X_mesh,
        Y_mesh,
        np.cos(configurations),
        np.sin(configurations),
        configurations,
        headaxislength=3,
        headwidth=4,
        scale=20,
        cmap="twilight",
        width=0.005,
    )
    return contour


fig, ax = plt.subplots(figsize=(11, 9))
# plt.figure()
contour = ax.quiver(
    X_mesh,
    Y_mesh,
    np.cos(configurations),
    np.sin(configurations),
    configurations,
    headaxislength=3,
    headwidth=4,
    scale=20,
    cmap="twilight",
    width=0.005,
)
plt.colorbar(contour)


animation = FuncAnimation(
    fig, update, frames=int(len(all_configurations) / 10), interval=2
)

plt.show()
