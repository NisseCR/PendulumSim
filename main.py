import pandas as pd

from pendulum import Pendulum


def export(df: pd.DataFrame):
    df.to_csv(path_or_buf='./results/log.txt', index=False, sep=';', header=True)


def main():
    pen = Pendulum(m2=2.0)
    df = pen.run(a1=2.0, a2=1.0, duration=20)
    export(df)
    print(df)


if __name__ == '__main__':
    main()
