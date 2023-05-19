#include <iostream>
#include <vector>
#include <tuple>
#include <ctime>
#include <cmath>
#include <fstream>

int *spins(int number)
{
    int *spin = new int[number * number];
    for (int i = 0; i < number * number; ++i)
    {
        spin[i] = 2 * (rand() % 2) - 1;
    }
    return spin;
}

int Single_Energy(int *spins, int number, int i, int j)
{
    int energy;
    int ind = number * i + j;
    int ni = number * ((number + i - 1) % number) + (number + j) % number;
    int nj = number * ((number + i) % number) + (number + j - 1) % number;
    int pi = number * ((number + i + 1) % number) + (number + j) % number;
    int pj = number * ((number + i) % number) + (number + j + 1) % number;
    energy = -spins[ind] * (spins[ni] + spins[nj] + spins[pi] + spins[pj]);
    return energy;
}

int Energy(int *spins, int number)
{
    int energy = 0;
    for (int i = 0; i < number; ++i)
    {
        for (int j = 0; j < number; ++j)
        {
            energy += Single_Energy(spins, number, i, j);
        }
    }
    return energy / 2;
}

std::tuple<int *, int> Monte_Carlo(int *spins, int total_energy, int number, double temperature)
{
    int i = rand() % number;
    int j = rand() % number;
    int prev_energy = Single_Energy(spins, number, i, j);
    int new_energy = -prev_energy;
    if ((double)rand() / RAND_MAX <= exp((prev_energy - new_energy) / temperature))
    {
        // std::cout << new_energy << std::endl;
        spins[number * i + j] = -spins[number * i + j];
        return {spins, total_energy - prev_energy + new_energy};
    }
    else
    {
        return {spins, total_energy};
    }
}

int main()
{
    srand(time(NULL));
    int number;
    double temperature;
    std::cin >> number;
    std::cin >> temperature;
    int *spin = spins(number);
    int ene = Energy(spin, number);
    std::cout << ene << std::endl; // Commented out unnecessary output

    long int iterations = 1000000000; // Use a constant for the loop count
    for (int i = 0; i < iterations; ++i)
    {
        auto [new_spin, new_ene] = Monte_Carlo(spin, ene, number, temperature);
        spin = new_spin;
        ene = new_ene;
    };

    std::cout << ene << std::endl;
    int sene = Energy(spin, number);
    std::cout << sene << std::endl;
    std::ofstream MyFile("spins.txt");
    for (int i = 0; i < number; ++i)
    {
        for (int j = 0; j < number; ++j)
        {
            MyFile << spin[number * i + j] << " ";
        }
        MyFile << std::endl;
    }
}
