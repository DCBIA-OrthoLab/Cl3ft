import SimpleITK as sitk
import numpy as np
import argparse


def main(args):

    print("Reading:", args.img)
    sitk_img = sitk.ReadImage(args.img)

    sitk_out = CorrectHisto(sitk_img, args.min_percentile, args.max_percentile, args.i_min, args.i_max)

    print("Writing:", args.out)

    sitk.WriteImage(sitk_out, args.out)

def CorrectHisto(sitk_img, min_percentile=0.01, max_percentile = 0.99, i_min=-1500, i_max=4000, num_bins=1000):

    print("Correcting scan contrast...")
    img_np = sitk.GetArrayFromImage(sitk_img)


    img_min = np.min(img_np)
    img_max = np.max(img_np)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = num_bins
    histo = np.histogram(img_np, definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_percentile, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_percentile, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min, i_min)
    res_max = min(res_max, i_max)
    

    img_np = np.where(img_np > res_max, res_max, img_np)
    img_np = np.where(img_np < res_min, res_min, img_np)

    output = sitk.GetImageFromArray(img_np)
    output.SetSpacing(sitk_img.GetSpacing())
    output.SetDirection(sitk_img.GetDirection())
    output.SetOrigin(sitk_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)
    
    return output

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Correct image histo')
    parser.add_argument('--img', type=str, help='Image to correct histogram', required=True)    
    parser.add_argument('--min_percentile', type=float, help='Minimum percentile', default=0.01)    
    parser.add_argument('--max_percentile', type=float, help='Maximum percentile', default=0.99)    
    parser.add_argument('--i_min', type=int, help='Maximum intensity', default=-1500)    
    parser.add_argument('--i_max', type=int, help='Maximum intensity', default=4000)    
    parser.add_argument('--num_bins', type=int, help='Maximum intensity', default=1000)    
    parser.add_argument('--out', type=str, help='Output image filename', default='out.nii.gz')

    args = parser.parse_args()

    main(args)
