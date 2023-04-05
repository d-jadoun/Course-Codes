import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants
import random

###################### Constants ##################################
PARTICLES_IN_ROW = 6
PARTICLES = PARTICLES_IN_ROW**3
EPSILON = 119.8 * constants.Boltzmann
SIGMA = 3.4 * constants.angstrom
MASS = 39.95 * 1.6747 * 1e-24  # gram
MASS_DENSITY = 0.5 * 1.68 * 1e6  # gram/meter^3
DISTANCE = (MASS / MASS_DENSITY) ** (1 / 3)  # meters
TEMPERATURE = 95  # Kelvin
MOVE_DISTANCE = 0.15 * SIGMA  # meters
BETA = 1 / (constants.Boltzmann * TEMPERATURE)
RUN = 1000000
BOX_SIZE = PARTICLES_IN_ROW * DISTANCE
VOLUME = BOX_SIZE**3
PRESSURE = 300000
######################################################################


def cubic_lattice(particles_in_row, initial_distance):
    """
    Construct a cubic lattice of particles.

    Parameters
    ----------
    particles_in_row : int
        Number of particles in one dimension of the lattice.
    initial_distance : float
        Separation distance between two adjacent particles.

    Returns
    -------
    numpy.ndarray
        An array of shape (particles_in_row ** 3, 3) representing the positions of the particles in the cubic lattice.
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
    This class calculates all kind of potentials required while running the Monte-Carlo simulations.
    From calculating individual potentials to two-dimensional arrays containing all the potentials,
    everything can be performed using this class.
    """

    def __init__(self, box_size, particles) -> None:
        """
        Initializes a PotentialForce object.

        Parameters:
        -----------
        box_size : float
            The size of the simulation box.
        particles : int
            The number of particles in the system.
        """
        self.box_size = box_size
        self.particles = particles

    def lennard_jones_potential(self, position_1, position_2):
        """
        Calculates the Lennard-Jones potential and force between two particles located at
        position_1 and position_2. The forces and potentials are calculated with minimum image
        convention for the purpose of the periodic boundary condition.

        Parameters:
        -----------
        position_1 : numpy.ndarray
            The position of the first particle.
        position_2 : numpy.ndarray
            The position of the second particle.

        Returns:
        --------
        potential : float
            The Lennard-Jones potential between the two particles.
        force : numpy.ndarray
            The force exerted on the first particle due to the second particle.
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
        Constructs an array with each element storing potentials between the particle "particle_number"
        and the rest of the particles in the system.

        Parameters:
        -----------
        particle_number : int
            The particle number for which potentials need to be calculated.
        positions : numpy.ndarray
            The positions of all the particles.

        Returns:
        --------
        potential_list : numpy.ndarray
            The list of potentials between the selected particle and all other particles.
        total_potential : float
            The total potential energy between the selected particle and all other particles.
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
        Constructs a two-dimensional potential array in which each element of the array stores the potential
        energy between two particles represented by the indices (i,j). A two-dimensional array for forces is
        constructed in the similar manner as well.

        Parameters:
        -----------
        positions : numpy.ndarray
            The positions of all the particles.

        Returns:
        --------
        potential_array : numpy.ndarray
            A two-dimensional array in which each element represents the potential energy between two particles
            represented by the indices (i,j).
        total_potential : float
            The total potential energy between all the particles.
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
        Takes the randomly moved particle position and updates the potentials.

        Parameters:
        -----------
        old_potential_matrix : numpy.ndarray
            The old two-dimensional potential array.
        new_potential_array : numpy.ndarray
            The new one-dimensional potential array for the selected particle.
        particle_number : int
            The particle number for which the potential array is updated.

        Returns:
        --------
        numpy.ndarray
            The updated two-dimensional potential array.
        """
        dummy = old_potential_matrix.copy()
        dummy[particle_number] = new_potential_array
        dummy[:, particle_number] = new_potential_array
        return dummy


class MonteCarlo:
    """
    We simulate the Grand Canonical Monte Carlo operation in this class.
    Particle movement or addition/deletion of particle is performed
    new energy and configurations are obtained.
    """

    def __init__(
        self, particles, move_distance, box_size, beta, volume, pressure
    ) -> None:
        self.particles = particles
        self.move_distance = move_distance
        self.box_size = box_size
        self.beta = beta
        self.volume = volume
        self.pressure = pressure

    def particle_operation(self, configurations):
        """
        Randomly selects a particle and either displaces it or adds/deletes it from the ensemble.

        Inputs:
            configurations: numpy array representing the positions of particles in a box.

        Returns:
            movement: bool -> True if a particle was moved, False otherwise.
            addition: bool -> True if a particle was added, False otherwise.
            deletion: bool -> True if a particle was deleted, False otherwise.
            old_energy: float -> energy of the old configuration.
            new_energy: float -> energy of the new configuration.
            new_configuration: numpy array -> new position of particles in 3D.
        """
        decider_1 = random.random()
        if decider_1 < 0.5:
            particle = random.randrange(0, self.particles, 1)
            initial_position = configurations[particle].copy()
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
            new_configuration = configurations.copy()
            new_configuration[particle] = new_position.copy()
            # print(new_configuration.shape, "decider_1")
            return True, False, False, particle, new_configuration
        else:
            decider_2 = random.random()
            if decider_2 < 0.5:
                particle = random.randrange(0, self.particles, 1)
                new_configuration = np.delete(configurations.copy(), particle, axis=0)
                # print(new_configuration.shape, "decider_2")
                return False, False, True, particle, new_configuration
            else:
                new_particle_position = 0 * configurations[1].copy()
                new_particle_position[0] = self.box_size * (random.random())
                new_particle_position[1] = self.box_size * (random.random())
                new_particle_position[2] = self.box_size * (random.random())
                new_configuration = np.concatenate(
                    (configurations.copy(), [new_particle_position])
                )
                # print(new_configuration.shape, "decider_3")
                return False, True, False, None, new_configuration

    def accept_reject(self, configurations):
        """
        Accepts or rejects the move based on the output of the previous function.

        Inputs:
            configurations: numpy array representing the positions of particles in a box.

        Returns:
            movement: bool -> True if a particle was moved, False otherwise.
            addition: bool -> True if a particle was added, False otherwise.
            deletion: bool -> True if a particle was deleted, False otherwise.
            old_energy: float -> energy of the old configuration.
            new_energy: float -> energy of the new configuration.
            new_configuration: numpy array -> new position of particles in 3D.
        """
        (
            displacement,
            addition,
            deletion,
            particle,
            new_configuration,
        ) = self.particle_operation(configurations)
        if displacement is True:
            new_energy = PotentialForce(self.box_size, self.particles).potential_array(
                particle, new_configuration
            )[1]
            old_energy = PotentialForce(self.box_size, self.particles).potential_array(
                particle, configurations
            )[1]
            if random.random() < np.exp(self.beta * (old_energy - new_energy)):
                return True, False, False, old_energy, new_energy, new_configuration
            else:
                return False, False, False, None, None, None
        elif addition is True:
            new_energy = PotentialForce(self.box_size, self.particles).potential_array(
                -1, new_configuration
            )[1]
            if random.random() < self.volume * self.beta * self.pressure * np.exp(
                -self.beta * new_energy
            ) / (self.particles + 1):
                return False, True, False, 0, new_energy, new_configuration
            else:
                return False, False, False, None, None, None
        elif deletion is True:
            old_energy = PotentialForce(self.box_size, self.particles).potential_array(
                particle, configurations
            )[1]
            if random.random() < self.particles * np.exp(self.beta * old_energy) / (
                self.volume * self.beta * self.pressure
            ):
                return False, False, True, old_energy, 0, new_configuration
            else:
                return False, False, False, None, None, None


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
total_particles = PARTICLES
total_particles_array = [PARTICLES]

# print(
#     MonteCarlo(PARTICLES, MOVE_DISTANCE, BOX_SIZE, BETA).particle_operation(
#         configurations
#     )
# )

# print(
#     MonteCarlo(
#         PARTICLES, MOVE_DISTANCE, BOX_SIZE, BETA, VOLUME, PRESSURE
#     ).accept_reject(configurations)[:3]
# )


while count < RUN:
    initiation = MonteCarlo(
        total_particles, MOVE_DISTANCE, BOX_SIZE, BETA, VOLUME, PRESSURE
    ).accept_reject(configurations)
    if initiation[0] is True:
        configurations = initiation[5]
        total_potential = total_potential + initiation[3] - initiation[4]
        total_particles_array.append(total_particles)
        total_potential_array.append(total_potential)
        print(count)
    elif initiation[1] is True:
        configurations = initiation[5]
        total_potential = total_potential + initiation[4]
        total_potential_array.append(total_potential)
        total_particles += 1
        total_particles_array.append(total_particles)
        print(count)
    elif initiation[2] is True:
        configurations = initiation[5]
        total_potential -= initiation[3]
        total_potential_array.append(total_potential)
        total_particles -= 1
        total_particles_array.append(total_particles)
        print(count)
    # print(configurations.shape)
    count += 1

plt.figure()
plt.plot(total_potential_array)

plt.figure()
plt.plot(total_particles_array)

plt.show()
