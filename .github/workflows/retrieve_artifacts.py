import wandb
import os
import click


@click.command()
@click.option('--registered_model_name')
@click.option('--target_dir', default='artifact_cache')
def main(registered_model_name, target_dir):
    wandb.init(project='cifar10-example', job_type='CI')
    model_art = wandb.use_artifact(registered_model_name)


    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    model_art.download(root=os.path.join(target_dir, "models"))


if __name__ == '__main__':
    main()
