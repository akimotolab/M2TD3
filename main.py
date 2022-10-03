import hydra

from trainer import Trainer


@hydra.main(config_path="configs", config_name="default")
def main(config):
    experiment_name = config["experiment_name"]
    trainer = Trainer(config, experiment_name)
    trainer.main()


if __name__ == "__main__":
    main()
