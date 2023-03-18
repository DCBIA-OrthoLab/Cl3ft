import SimpleITK as sitk
import numpy as np
import argparse
import os
import glob
import sys
import pandas as pd




def main(args):



	image_info = []
	for fn in glob.glob(os.path.join(args.dir, args.ext)):

		print("Reading:", fn)
		img = sitk.ReadImage(fn)
		spc = img.GetSpacing()
		size = img.GetSize()

		image_info.append({"img": fn, "spc_x": spc[0], "spc_y": spc[1], "spc_z": spc[2], "size_x": size[0], "size_y": size[1], "size_z": size[2]})
	

	df = pd.DataFrame(image_info)
	df.to_csv(args.out, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image information', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, help='Directory to get image information')
    parser.add_argument('--ext', type=str, help='Extension type')
    parser.add_argument('--out', type=str, help='Ouput csv')
    
    args = parser.parse_args()

    main(args)
