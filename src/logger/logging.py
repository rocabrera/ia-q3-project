import os
import logging


logging.basicConfig(format='%(asctime)s %(levelname)-s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO,
                    filename=os.path.join('src','logger','trained_models.log'))

logger = logging.getLogger('my_app')


