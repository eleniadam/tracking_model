import click
from time import time
from .relabel import *
from .superimpose import *
from .model import *
import os
from . import __version__


@click.group()
@click.version_option(__version__)
def cli():
    pass


@cli.command()
@click.option('--tree_file',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="CSV file with lineage tree information.")
@click.option('--image_i',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="File of 3D image timestamp i.")
@click.option('--image_ii',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="File of 3D image timestamp i+1.")
@click.option('--output_dir',required=True,
              type=click.Path(exists=True,dir_okay=True,readable=True),
              help="Relabelled images output directory.")
def generate_relabelledImage(tree_file, image_i, image_ii, output_dir):
    click.echo('Invoking Relabelling...')
    t0 = time()
    relimage_dir = run_relabelling(tree_file, image_i, image_ii, output_dir)
    t1 = time() - t0
    click.echo('Relabelled images generated here:' + relimage_dir)
    click.echo("Time elapsed: " + str(t1))

@cli.command()
@click.option('--image_i',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="File of 3D image timestamp i.")
@click.option('--image_ii',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="File of 3D image timestamp i+1.")
@click.option('--output_dir',required=True,
              type=click.Path(exists=True,dir_okay=True,readable=True),
              help="Superimposed images output directory.")
def generate_superimposedImage(image_i, image_ii, output_dir):
    click.echo('Invoking Superimpose...')
    t0 = time()
    supimage_dir = run_superimpose(image_i, image_ii, output_dir)
    t1 = time() - t0
    click.echo('Superimposed images generated here:' + supimage_dir)
    click.echo("Time elapsed: " + str(t1))

@cli.command()
@click.option('--original_dir',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="Original unlabelled images directory.")
@click.option('--groundtruth_dir',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="Labelled images directory.")
@click.option('--output_dir',required=True,
              type=click.Path(exists=True,dir_okay=True,readable=True),
              help="Model output directory.")
def train_model(original_dir, groundtruth_dir, output_dir):
    click.echo('Invoking Train...')
    t0 = time()
    tmodel_dir = run_train_model(original_dir, groundtruth_dir, output_dir)
    t1 = time() - t0
    click.echo('Trained model information saved here:' + tmodel_dir)
    click.echo("Time elapsed: " + str(t1))

@cli.command()
@click.option('--image',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="Original unlabelled image.")
@click.option('--model_file',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="Trained model file.")
@click.option('--weights_file',required=True,
              type=click.Path(exists=True,file_okay=True,readable=True),
              help="Optimal model weights file.")
@click.option('--output_dir',required=True,
              type=click.Path(exists=True,dir_okay=True,readable=True),
              help="Predicted image output directory.")
def predict_image(image, model_file, weights_file, output_dir):
    click.echo('Invoking Predict...')
    t0 = time()
    pmodel_dir = run_predict_model(image, model_file, weights_file, output_dir)
    t1 = time() - t0
    click.echo('Predicted image saved here:' + pmodel_dir)
    click.echo("Time elapsed: " + str(t1))

def main():
    cli()


