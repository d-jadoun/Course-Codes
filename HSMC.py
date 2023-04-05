import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants
import random

###################### Constants ##################################
PARTICLES_IN_ROW = 5
PARTICLES = PARTICLES_IN_ROW**3
RADIUS = 0.1
MOVE_DISTANCE = 1  # meters
RUN = 5000000
DISTANCE = 1
BOX_SIZE = DISTANCE * PARTICLES_IN_ROW
VOLUME = BOX_SIZE**3
NBINS = 300
######################################################################


def cubic_lattice(particles_in_row, initial_distance):
    """
    Constructs a cubic lattice with the specified number of particles in a single
    dimension and the initial distance between two particles.

    Args:
    particles_in_row (int): Number of particles in a single dimension of the cubic lattice.
    initial_distance (float): Initial distance between two neighboring particles.

    Returns:
    np.ndarray: A 3D numpy array containing the coordinates of each particle in the cubic lattice.
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


def radial_dist_fun(configuration, n_bins):
    """
    Calculates the radial distribution function for a given configuration of the system
    using a specified number of bins.

    Args:
    configuration (np.ndarray): A 3D numpy array containing the coordinates of each particle
                                in the system.
    n_bins (int): Number of bins for the radial distribution function.

    Returns:
    Tuple: A tuple containing two numpy arrays.
        - g (np.ndarray): Radial distribution function values for each bin.
        - rs (np.ndarray): Array of radii for each bin.
    """

    dr = BOX_SIZE / n_bins
    rs = dr * np.arange(1, n_bins + 1)
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
    A class for performing Monte Carlo simulations on a system of particles.

    Attributes:
    -----------
    particles : int
        The number of particles in the system.
    move_distance : float
        The maximum distance that a particle can move in a trial move.
    box_size : float
        The size of the simulation box.
    radius : float
        The radius of the particles.

    Methods:
    --------
    trial_move(particle, new_configuration):
        Determines whether a proposed move is valid.

    accept_reject(configuration):
        Performs a trial move and decides whether to accept or reject it.
    """

    def __init__(self, particles, move_distance, box_size, radius):
        """
        Initializes a MonteCarlo object.

        Parameters:
        -----------
        particles : int
            The number of particles in the system.
        move_distance : float
            The maximum distance that a particle can move in a trial move.
        box_size : float
            The size of the simulation box.
        radius : float
            The radius of the particles.
        """
        self.particles = particles
        self.move_distance = move_distance
        self.box_size = box_size
        self.radius = radius

    def trial_move(self, particle, new_configuration):
        """
        Determines whether a proposed move is valid.

        Parameters:
        -----------
        particle : int
            The index of the particle to move.
        new_configuration : numpy.ndarray
            An array containing the proposed new positions of all particles.

        Returns:
        --------
        valid_move : bool
            True if the proposed move is valid, False otherwise.
        """
        particle_position = new_configuration[particle]
        vectors = new_configuration - particle_position
        vectors -= np.round(vectors / self.box_size) * self.box_size
        distance = np.linalg.norm(vectors, axis=1)
        invalid_pos = distance < 2 * self.radius
        if invalid_pos.sum() == 1:
            return True
        else:
            return False

    def accept_reject(self, configuration):
        """
        Performs a trial move and decides whether to accept or reject it.

        Parameters:
        -----------
        configuration : numpy.ndarray
            An array containing the current positions of all particles.

        Returns:
        --------
        accept : bool
            True if the trial move is accepted, False otherwise.
        new_configuration : numpy.ndarray
            An array containing the new positions of all particles if the move is accepted,
            or the current positions if the move is rejected.
        """
        particle = random.randrange(0, self.particles, 1)
        initial_position = configuration[particle].copy()
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
        new_configuration = configuration.copy()
        new_configuration[particle] = new_position.copy()
        trial = self.trial_move(particle, new_configuration)
        if trial is True:
            return True, new_configuration
        if trial is False:
            return False, configuration


count = 0

INITIAL_POSITIONS = cubic_lattice(
    particles_in_row=PARTICLES_IN_ROW, initial_distance=DISTANCE
)
configurations = INITIAL_POSITIONS.copy()

configurations_array = []

while count < RUN:
    initiation = MonteCarlo(PARTICLES, MOVE_DISTANCE, BOX_SIZE, RADIUS).accept_reject(
        configurations
    )
    if initiation[0] is True:
        configurations = initiation[1]
        if count > RUN - int(RUN / 10):
            configurations_array.append(configurations)

    count += 1


y, x = radial_dist_fun(INITIAL_POSITIONS, NBINS)
plt.figure()
plt.plot(x, y)

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
fin_rdf = np.sum(rdf_array, axis=0) / np.shape(rdf_array)[0]
plt.figure()
plt.plot(x, fin_rdf)
plt.show()
