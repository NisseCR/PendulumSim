import random
from typing import List, Tuple


class Generator:
    def __init__(self):
        pass

    @staticmethod
    def generate_angles(sample_size: int) -> List[Tuple[int, float, float]]:
        samples = []

        id = 0
        for n in range(sample_size):
            a1 = random.uniform(0.0, 2.0)
            a2 = random.uniform(0.0, 2.0)
            samples.append((id, a1, a2))
            id += 1

        return samples

    @staticmethod
    def generate_samples(mass_lower: float, mass_upper: float, interval: float, angles: List[Tuple[int, float, float]])\
            -> List[Tuple[float, List[Tuple[int, float, float]]]]:
        samples = []

        mass = mass_lower
        while mass <= mass_upper:
            samples.append((mass, angles))
            mass += interval

        return samples

    def generate(self, sample_size: int, mass_lower: float, mass_upper: float, interval: float) \
            -> List[Tuple[float, List[Tuple[int, float, float]]]]:
        angles = self.generate_angles(sample_size)
        samples = self.generate_samples(mass_lower, mass_upper, interval, angles)
        return samples
