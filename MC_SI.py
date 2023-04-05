import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants
import random

###################### Constants ##################################
PARTICLES_IN_ROW = 5
PARTICLES = PARTICLES_IN_ROW**3
EPSILON = 119.8 * constants.Boltzmann
SIGMA = 3.4 * constants.angstrom
MASS = 39.95 * 1.6747 * 1e-24  # gram
MASS_DENSITY = 1.1 * 1.68 * 1e6  # gram/meter^3
DISTANCE = (MASS / MASS_DENSITY) ** (1 / 3)  # meters
TEMPERATURE = 95  # Kelvin
MOVE_DISTANCE = 0.15 * SIGMA  # meters
BETA = 1 / (constants.Boltzmann * TEMPERATURE)
RUN = 500000
BOX_SIZE = PARTICLES_IN_ROW * DISTANCE
VOLUME = BOX_SIZE**3
NBINS = 1000
######################################################################


def cubic_lattice(particles_in_row, initial_distance):
    """
    Construct a cubic lattice of particles with the given number of particles
    in each row and the initial distance between two adjacent particles.

    Args:
        particles_in_row: The number of particles in a single dimension of the cubic lattice.
        initial_distance: The initial separation distance between two adjacent particles.

    Returns:
        A numpy array of shape (N, 3) containing the position of each particle in the cubic lattice.
        The `N` is the total number of particles, which is equal to `particles_in_row**3`.
    """
    number_of_particles = particles_in_row**3
    position = np.zeros([number_of_particles, 3])
    count_0 = 0
    for i in range(particles_in_row):
        for j in range(particles_in_row):
            for k in range(particles_in_row):
                position[count_0] = (
                    initial_distance * i,
                    initial_distance * j,
                    initial_distance * k,
                )
                count_0 += 1
    return position


class PotentialForce:
    """
    This class provides methods to calculate various potentials required for Monte Carlo simulations,
    including individual potentials and two-dimensional arrays containing all potentials.
    """

    def __init__(self, box_size, particles) -> None:
        """
        Initializes a PotentialForce object.

        Args:
        box_size (float): The size of the simulation box.
        particles (int): The number of particles in the system.
        """
        self.box_size = box_size
        self.particles = particles

    def lennard_jones_potential(self, position_1, position_2):
        """
        Calculates the Lennard-Jones potential and force between two particles located at position_1 and position_2,
        taking into account the periodic boundary conditions.

        Args:
        position_1 (array): The position of the first particle.
        position_2 (array): The position of the second particle.

        Returns:
        tuple: A tuple containing the potential energy and force between the two particles.
        """
        distance_x = (
            position_1[0]
            - position_2[0]
            - np.rint((position_1[0] - position_2[0]) / self.box_size) * self.box_size
        )  # Minimum image convention implementation
        distance_y = (
            position_1[1]
            - position_2[1]
            - np.rint((position_1[1] - position_2[1]) / self.box_size) * self.box_size
        )
        distance_z = (
            position_1[2]
            - position_2[2]
            - np.rint((position_1[2] - position_2[2]) / self.box_size) * self.box_size
        )
        distance = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        potential = 4 * EPSILON * ((SIGMA / distance) ** 12 - (SIGMA / distance) ** 6)
        force = (
            48
            * EPSILON
            * np.array([distance_x, distance_y, distance_z])
            * ((SIGMA / distance) ** 12 - 0.5 * (SIGMA / distance) ** 6)
            / distance**2
        )
        return potential, force

    def potential_array(self, particle_number, positions):
        """
        Constructs an array that stores the potentials between the particle represented by particle_number
        and the rest of the particles in the system.

        Args:
        particle_number (int): The index of the particle for which the potentials are being calculated.
        positions (array): The array of particle positions.

        Returns:
        tuple: A tuple containing an array of potentials and the total potential energy between the particle and the rest of the system.
        """
        potential_list = np.zeros([self.particles])
        for i in range(self.particles):
            if i == particle_number:
                potential_list[i] = 0
            else:
                potential_list[i] = self.lennard_jones_potential(
                    positions[particle_number], positions[i]
                )[0]
        return potential_list, np.sum(potential_list)

    def potential_matrix(self, positions):
        """
        Constructs a two-dimensional array that stores the potential energy between each pair of particles
        represented by the indices (i,j). Also constructs a two-dimensional array for the forces between each
        pair of particles in the same manner.

        Args:
        positions (array): The array of particle positions.

        Returns:
        tuple: A tuple containing the two-dimensional array of potentials and the total potential energy between all pairs of particles.
        """
        force_array = np.zeros([self.particles, self.particles, 3])
        potential_array = np.zeros([self.particles, self.particles])
        for i in range(0, self.particles - 1):
            for j in range(i + 1, self.particles):
                potential_array[i, j] = self.lennard_jones_potential(
                    positions[i], positions[j]
                )[0]
                potential_array[j, i] = potential_array[i, j]
                force_array[i, j] = self.lennard_jones_potential(
                    positions[i], positions[j]
                )[1]
                force_array[j, i] = force_array[i, j]
        return (
            potential_array,
            np.sum(np.reshape(potential_array, self.particles * self.particles)) / 2,
        )

    def potential_update(
        self, old_potential_matrix, new_potential_array, particle_number
    ):
        """
        Update the potentials in the given potential matrix by replacing the row and column
        corresponding to the specified particle number with the new potential array.

        Parameters:
        -----------
        old_potential_matrix : numpy.ndarray
            The current potential matrix to be updated.
        new_potential_array : numpy.ndarray
            The new array of potentials for the moved particle.
        particle_number : int
            The index of the moved particle in the potential matrix.

        Returns:
        --------
        numpy.ndarray
            The updated potential matrix.
        """
        dummy = old_potential_matrix.copy()
        dummy[particle_number] = new_potential_array
        dummy[:, particle_number] = new_potential_array
        return dummy


def radial_dist_fun(configuration, n_bins):
    """
    Calculate the radial distribution function (RDF) of the system given the configuration
    of the particles and the number of bins n_bins.

    Parameters:
    -----------
    configuration : np.ndarray
        The positions of the particles in the system.
    n_bins : int
        The number of bins to use in the RDF calculation.

    Returns:
    --------
    g : np.ndarray
        The radial distribution function values for each bin.
    rs : np.ndarray
        The bin distances.
    """
    dr = BOX_SIZE / n_bins
    rs = dr * np.arange(1, n_bins + 1)  # np.array(dr*i for i = 1:n_bins+1)
    h = np.zeros(n_bins)
    for i in range(PARTICLES):
        for j in range(i + 1, PARTICLES):
            ij_vector = configuration[i] - configuration[j]
            ij_vector -= np.round(ij_vector / BOX_SIZE) * BOX_SIZE
            ij_distance = np.linalg.norm(ij_vector)
            h[int(ij_distance / dr)] += 1
    g = (VOLUME / (PARTICLES**2 * 2 * np.pi * dr)) * h / rs**2
    return g, rs


class MonteCarlo:
    """
    Metropolis Monte-Carlo simulations are performed with this class.
    The Metropolis scheme helps find out the ground state of a system of particles.
    Under this scheme, we move a random particle and check if the energy
    decreases or not. We continue to do so until we reach the equilibrium.
    """

    def __init__(self, particles, move_distance, box_size, beta) -> None:
        """
        Initialize the Monte Carlo simulation with the number of particles, move distance, box size,
        and inverse temperature beta.

        Args:
            particles (int): The number of particles in the system.
            move_distance (float): The maximum distance a particle can be moved in one step.
            box_size (float): The size of the simulation box.
            beta (float): The inverse temperature of the system.
        """
        self.move_distance = move_distance
        self.particles = particles
        self.box_size = box_size
        self.beta = beta

    def pick_move(self, positions):
        """
        Randomly pick a particle and move it by a random distance. Return the index of the moved
        particle and the new configuration of the system.

        Args:
            positions (np.ndarray): The positions of the particles in the system.

        Returns:
            Tuple[int, np.ndarray]: The index of the moved particle and the new configuration of
            the system.
        """
        particle = random.randrange(0, self.particles, 1)
        initial_position = positions[particle].copy()
        new_position = 0 * initial_position
        new_position[0] = initial_position[0] + self.move_distance * (
            random.random() - 0.5
        )
        new_position[1] = initial_position[1] + self.move_distance * (
            random.random() - 0.5
        )
        new_position[2] = initial_position[2] + self.move_distance * (
            random.random() - 0.5
        )
        new_configuration = positions.copy()
        new_configuration[particle] = new_position.copy()
        return particle, new_configuration

    def accept_reject(self, positions):
        """
        Acquire the changed configuration after displacing a random particle and check if the
        energy of the system changes. Return whether the move was accepted, the index of the moved
        particle, the new potential energy, the old potential energy, and the new configuration of
        the system.

        Args:
            positions (np.ndarray): The positions of the particles in the system.

        Returns:
            Tuple[bool, int, float, float, np.ndarray]: Whether the move was accepted, the index of
            the moved particle, the new potential energy, the old potential energy, and the new
            configuration of the system.
        """
        particle, new_positions = self.pick_move(positions)
        new_potential = PotentialForce(self.box_size, self.particles).potential_array(
            particle, new_positions
        )[1]
        old_potential = PotentialForce(self.box_size, self.particles).potential_array(
            particle, positions
        )[1]
        if random.random() < np.exp(self.beta * (old_potential - new_potential)):
            return True, particle, new_potential, old_potential, new_positions
        else:
            return False, particle, old_potential, positions


count = 0

INITIAL_POSITIONS = cubic_lattice(
    particles_in_row=PARTICLES_IN_ROW, initial_distance=DISTANCE
)
INITIAL_POTENTIAL_MATRIX, INITIAL_TOTAL_POTENTIAL = PotentialForce(
    BOX_SIZE, PARTICLES
).potential_matrix(INITIAL_POSITIONS)

potential_matrix = INITIAL_POTENTIAL_MATRIX.copy()
total_potential = INITIAL_TOTAL_POTENTIAL.copy()
total_potential_array = [total_potential]
configurations = INITIAL_POSITIONS.copy()
configurations_array = []

while count < RUN:
    initiation = MonteCarlo(PARTICLES, MOVE_DISTANCE, BOX_SIZE, BETA).accept_reject(
        configurations
    )
    if initiation[0] == True:
        configurations = initiation[4]
        total_potential = total_potential + initiation[2] - initiation[3]
        total_potential_array.append(total_potential)
        if count > RUN - int(RUN / 10):
            configurations_array.append(configurations)
    count += 1

final_runs = np.shape(np.array(configurations_array))[0]
print(final_runs)

rdf_array = []

for i in range(
    0,
    final_runs,
    200,
):
    rdf_array.append(radial_dist_fun(configurations_array[i], NBINS)[0])


rdf_array = np.array(rdf_array)


plt.plot(total_potential_array)
plt.show()

y, x = radial_dist_fun(INITIAL_POSITIONS, NBINS)
plt.figure()
plt.plot(x, y)


fin_rdf = np.sum(rdf_array, axis=0) / np.shape(rdf_array)[0]
plt.figure()
plt.plot(x, fin_rdf)
plt.show()
