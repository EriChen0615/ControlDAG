import click
import shutil
import os
import pathlib
import runway_main as rw_main 

@click.group()
def cli():
    pass

@click.command()
@click.option('--dest', '-d', 'dest_dir',
    required=True, 
    type=str,
    default=os.getcwd())
def init(dest_dir):
    assert not os.path.exists('configs/'), """
        configs/ folder exist. You may be overwriting an already initialized
        runway project. Rename or delete configs/ folder and try again if you are sure.
        """
    click.echo("Initializing...")
    runway_dir = pathlib.Path(__file__).parent.resolve() # get parent directory
    click.echo(runway_dir)
    shutil.copytree(os.path.join(runway_dir, 'configs'), os.path.join(dest_dir, 'configs/'))
    # shutil.copytree(os.path.join(runway_dir, 'code_templates'), ".")
    click.echo("Project initialized. Welcome to Runway!")

# @click.command()
# @click.option('--config', '-c', 'config_file',
#     required=True,
#     type=str)
# def inspect_data(config_file):
#     assert os.path.exists(config_file), """
#         The specified configuration file does not exist!
#     """
#     click.echo(f"Inspecting data pipeline defined in {config_file}")
#     test_data_main(config_file)

@click.command()
@click.option('--config', '-c', 'config_file',
    required=True,
    type=str)
def prepare_data(config_file):
    assert os.path.exists(config_file), """
        The specified configuration file does not exist!
    """
    click.echo(f"Preparing data as defined in {config_file}")
    rw_main.prepare_data()
    click.echo(f"Data prepared!")
   



@click.command('hi')
def hi():
    click.echo("hello from runway!")

cli.add_command(init)
cli.add_command(hi)
cli.add_command(prepare_data)

if __name__ == '__main__':
    cli()
    
