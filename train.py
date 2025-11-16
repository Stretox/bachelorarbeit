import argparse
import os
from training.training_loop import train_hairflownet


def main(args):

    print(f'Args:{args} \n')

    if args.occ:
        train_hairflownet(args.hair_folder, args.camera_folder, args.checkpoint ,args.multi)


    print('Training Finished!!!')


if __name__ == "__main__":

    hair_folder = "data/hairinputdata" 
    camera_folder = "data/cameras"  

    parser = argparse.ArgumentParser(description="Train the Models")
    parser.add_argument("hair_folder", nargs='?', default=hair_folder, help="Path to the folder containing the Hair Strands")
    parser.add_argument("-c","--checkpoint", help="Path to the folder containing the Hair Strands")
    parser.add_argument("camera_folder", nargs='?', default=camera_folder, help="Path to the folder taht contains the camera.json")
    parser.add_argument('-b', '--backbone', action='store_true', help="Run Backbone Training")
    parser.add_argument('-o', '--occ', action='store_true', help="Run OccViT Training")
    parser.add_argument('-s', '--san', action='store_true', help="Sanity Checking code")
    parser.add_argument('-m', '--multi', action='store_true', help="Activate Multi Processing")
    parser.add_argument('-d', '--directory', default="", help="Checkpoint to be loaded")
    parser.add_argument('-mini', '--small', action='store_true', help="Skip ")
    args = parser.parse_args()


    main(args)