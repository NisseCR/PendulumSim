from typing import List

import pandas as pd

from pendulum import Pendulum
from generator import Generator


def export(df: pd.DataFrame, name: str):
    df.to_csv(path_or_buf=f"./results/{name}.txt", index=False, sep=';', header=True)


def analyse(dfs: List[pd.DataFrame]):
    df = pd.concat(dfs)
    df['v1_abs'] = df['v1'].abs()
    df['v2_abs'] = df['v2'].abs()
    df = df.groupby('m2').agg({'v1_abs': 'mean', 'v2_abs': 'mean'})
    export(df, 'speeds')
    print(df)


def main():
    gen = Generator()
    samples = gen.generate(sample_size=1, mass_lower=0, mass_upper=100, interval=100)

    dfs = []
    for mass, angles in samples:
        for identifier, a1, a2 in angles:

            # Simulate
            pen = Pendulum(identifier, mass)
            df = pen.run(a1, a2, duration=40)
            export(df, f"m{mass}_id{identifier}")

            # Analyse
            print(f"#{identifier} Mass: {mass} | Angles: ({a1}, {a2})")
            # print(df)
            dfs.append(df)

    analyse(dfs)


if __name__ == '__main__':
    main()
