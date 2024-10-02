import wandb

#Log to Wandb
class Logger:
    def __init__(self, experiment_name, project='FER_Thesis', logger_name='logger', dir=None):
        logger_name = f'{logger_name}-{experiment_name}'
        self.logger = wandb.init(project=project, name=logger_name, dir=dir, reinit=False)
    
    def get_logger(self):
        return self.logger