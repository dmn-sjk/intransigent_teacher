import shutil
import logging
import sys
import wandb

logger = logging.getLogger(__name__)


def clear_on_exit(res_dir, log_wandb, wandb_dir):
    folders_removed = False
    try:
        shutil.rmtree(res_dir)
        if log_wandb:
            wandb.finish()
            shutil.rmtree(wandb_dir)
        folders_removed = True
    except FileNotFoundError:
        folders_removed = True
    
    if folders_removed:
        logger.info("Logs removed.")
    else:
        logger.info("Problem with removing logs.")

    sys.exit()