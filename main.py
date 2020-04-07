import click

import scene1


@click.command()
@click.option("--scene", "-s", type=int)
def main(scene):
    if scene == 1:
        scene1.run()


if __name__ == "__main__":
    main()
