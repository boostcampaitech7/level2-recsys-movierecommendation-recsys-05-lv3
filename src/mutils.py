
import os



def check(output_dir) :
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
