import pandas as pd

from pendulum import Pendulum
from generator import Generator


def export(df: pd.DataFrame, name: str):
    df.to_csv(path_or_buf=f"./results/{name}.txt", index=True, sep=';', header=True)


def main():
    gen = Generator()
    samples = gen.generate(sample_size=5, mass_lower=1.0, mass_upper=2.0, interval=1.0)

    results = []
    for mass, angles in samples:
        for identifier, a1, a2 in angles:

            # Simulate
            pen = Pendulum(identifier, mass)
            df = pen.run(a1, a2, duration=20)
            export(df, f"m{mass}_id{identifier}")

            # Analyse
            print(f"#{identifier} Mass: {mass} | Angles: ({a1}, {a2})")
            results.append(df)


if __name__ == '__main__':
    main()
