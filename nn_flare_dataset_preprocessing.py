import glob
import json
import os
import os.path as osp
import shutil
import cv2
import copy

import imageio as io
import numpy as np
import pandas as pd
import tqdm

from pathlib import Path

FRAGMENT_SIZE = 512
PADDING = 0
FRAGMENT_OVERLAP_RATIO = 0.2
DATA_SPLITS = ['test', 'train', 'valid']
DS = ['valid']
OUTPUT_SIZE = 512


DEFAULT_NJOBS = -1

        


def slice_axis(axis_length, fragment_size, overlap_ratio):
    overlap = fragment_size * overlap_ratio
    num_centers = 1
    centers = []

    if num_centers == 1:
        centers.append(np.random.randint(fragment_size / 2,
                                         axis_length - fragment_size / 2 + 1))
    elif num_centers > 1:
        min_center = fragment_size / 2
        max_center = axis_length - fragment_size / 2

        centers.append(np.random.randint(min_center,
                                         max_center - (num_centers - 1) *
                                         (fragment_size - overlap) + 1))

        for i in range(1, num_centers - 1):
            centers.append(np.random.randint(centers[-1] + overlap,
                                             max_center - (num_centers - i - 1) * \
                                             (fragment_size - overlap) + 1))

        centers.append(np.random.randint(centers[-1] + (fragment_size - overlap),
                                         max_center + 1))
    return centers


def generate_crop_centers(img_size, fragment_size, overlap_ratio): 
    height, width = img_size[:2]
    if height < fragment_size or width < fragment_size:
        return []

    center_coords = []

    y_s = slice_axis(height, fragment_size, overlap_ratio)
    for y in y_s:
        x_s = slice_axis(width, fragment_size, overlap_ratio)
        for x in x_s:
            center_coords.append((y, x))
    return center_coords

    
def crop(img, rect): 
    x_left, y_top, w, h = rect
    return img[y_top:y_top+h, x_left:x_left+w]




def create_glare(inp_img):
    assert inp_img.dtype != np.uint8 or inp_img.dtype != np.uint16
    depth = 255 if inp_img.dtype == np.uint8 else 2 ** 16 - 1

    ##########
    # prepare clear images
    res_image = copy.deepcopy(inp_img).astype(np.int64)
    res_image_gt = copy.deepcopy(inp_img).astype(np.int64)
    ##########
    
    
    
    ##########
    # make glare
    glare_needed = True
    
    glare = np.zeros((FRAGMENT_SIZE, FRAGMENT_SIZE, 3))
    
    cen_x = np.random.randint(0, FRAGMENT_SIZE)
    cen_y = np.random.randint(0, FRAGMENT_SIZE)
    
    r_gl = np.random.randint(200, 255)
    g_gl = np.random.randint(0, 204)
    b_gl = np.random.randint(0, 100)
    
    
    mask_gl = np.ones((FRAGMENT_SIZE, FRAGMENT_SIZE, 1))
    
    
    inv_low = 0.
    inv_high = 1.#np.random.uniform(0.8, 1.)
    
    
    r = np.zeros((4))
    r[0] = (cen_x)**2 + (cen_y)**2
    r[1] = (cen_x)**2 + (FRAGMENT_SIZE - cen_y)**2
    r[2] = (FRAGMENT_SIZE - cen_x)**2 + (cen_y)**2
    r[3] = (FRAGMENT_SIZE - cen_x)**2 + (FRAGMENT_SIZE - cen_y)**2
    rad = np.sqrt(np.amax(r)) 
    direction = np.argmax(r)

    
    crad = np.random.uniform(0.03, 0.09) * rad # fonaric rad
    
    frad = rad#np.random.uniform(0.5, 1.2) * rad # fading rad
    fade_power = np.random.uniform(0.8, 1.2)
    color_fade_power_g = np.random.uniform(1., 1.5)
    color_fade_power_b = np.random.uniform(1., 1.5)
    ###########
    
    
    
    ###########
    #compute distance matrix            
    
    dist_x = np.ones((FRAGMENT_SIZE, FRAGMENT_SIZE))
    mult = np.array(range(FRAGMENT_SIZE))
    dist_x = np.multiply(dist_x, mult)
    dist_y = np.multiply(dist_x, np.ones((FRAGMENT_SIZE, FRAGMENT_SIZE))).T
    
    dist_x -= cen_x
    dist_y -= cen_y
    

    distances = np.sqrt(np.multiply(dist_x, dist_x) + np.multiply(dist_y, dist_y))
    distances[distances < crad] = crad
    distances[distances > frad] = frad
    ##########
    
    
    ##########
    #create mask and glare
    mask_gl[:, :, 0] = inv_high - ((distances - crad)/(frad - crad))**(fade_power) * (inv_high - inv_low)

    
    glare[:, :, 0] = (255 - ((distances - crad) / (frad - crad))**(color_fade_power_g) * (255 - r_gl)).astype(int)
    glare[:, :, 1] = (255 - ((distances - crad) / (frad - crad))**(color_fade_power_g) * (255 - g_gl)).astype(int)
    glare[:, :, 2] = (255 - ((distances - crad) / (frad - crad))**(color_fade_power_b) * (255 - b_gl)).astype(int)
    ##########
    
   
    ##########
    #create glared image V
    if(glare_needed):
        glare = np.multiply(glare, (mask_gl))
        
        res_image = np.multiply(res_image,  1 - mask_gl)
        res_image += glare
    else:
        res_image = np.multiply(res_image, np.ones((FRAGMENT_SIZE, FRAGMENT_SIZE, 1)))
        cv2.circle(glare, (cen_x, cen_y), crad.astype(int), (255, 255, 255), -1)
        res_image += glare
    ##########
    
    
    
    ##########
    #create ground truth image GT
    if(glare_needed):
        mask_gt = mask_gl
        mask_gt[mask_gt < inv_high] = 0.
    
        glare_gt = 255*np.ones((FRAGMENT_SIZE, FRAGMENT_SIZE, 3))#glare
        glare_gt = np.multiply(glare_gt, (mask_gt))
        res_image_gt = np.multiply(res_image_gt, 1 - mask_gt)
        
        res_image_gt += glare_gt
    else:
        res_image_gt = np.multiply(res_image_gt, mask_gl)
        res_image_gt += glare
    ##########
    
    
    res_image = np.minimum(res_image, 255.)
    res_image_gt = np.minimum(res_image_gt, 255.)
   

    return res_image, res_image_gt





def flare_mixup_wrapper(filename, split, output_path):

    img = io.imread(filename)

    if len(img.shape) == 2 or img.shape[2] != 3:
        return 

    if min(img.shape[:-1]) < (FRAGMENT_SIZE + PADDING * 2): 
        return 
    
    center_coords = []
    if split == 'test':
        center_coords = [(int(img.shape[0] / 2) - PADDING,
                          int(img.shape[1] / 2) - PADDING)]
    else:
        center_coords =  generate_crop_centers((img.shape[0] - 2*PADDING,
                                                img.shape[1] - 2*PADDING),
                                                FRAGMENT_SIZE,
                                                FRAGMENT_OVERLAP_RATIO)   

    if not len(center_coords):
        return

    crop_rects = [(cc[1] - int(FRAGMENT_SIZE/2) + PADDING, 
                   cc[0] - int(FRAGMENT_SIZE/2) + PADDING,
                   FRAGMENT_SIZE, FRAGMENT_SIZE) for cc in center_coords]

    basename = osp.basename(filename).split('.')[0] 

    for i, crop_rect in enumerate(crop_rects):
        flare_x, flare_y = create_glare(
            crop(img, crop_rect)
        )

        basename_x = '{}.00.jpg'.format(Path(filename).stem, i)
        basename_y = '{}.00.jpg'.format(Path(filename).stem, i)
        
        
        flare_x = np.uint8(flare_x)
        flare_y = np.uint8(flare_y)
        
        io.imwrite(osp.join(output_path, 'x_' + split, basename_x), flare_x)
        io.imwrite(osp.join(output_path, 'y_' + split, basename_y), flare_y)
        
        
    
    if(FRAGMENT_SIZE != 256):
        center_coords = []
        if split == 'test':
            center_coords = [(int(img.shape[0] / 2) - PADDING,
                              int(img.shape[1] / 2) - PADDING)]
        else:
            center_coords =  generate_crop_centers((img.shape[0] - 2*PADDING,
                                                    img.shape[1] - 2*PADDING),
                                                    FRAGMENT_SIZE,
                                                    FRAGMENT_OVERLAP_RATIO)   

        if not len(center_coords):
            return

        crop_rects = [(cc[1] - int(FRAGMENT_SIZE/2) + PADDING, 
                       cc[0] - int(FRAGMENT_SIZE/2) + PADDING,
                       FRAGMENT_SIZE, FRAGMENT_SIZE) for cc in center_coords]

        basename = osp.basename(filename).split('.')[0] 

        for i, crop_rect in enumerate(crop_rects):
            flare_x, flare_y = create_glare(
                crop(img, crop_rect)
            )

            basename_x = '{}.01.jpg'.format(Path(filename).stem, i)
            basename_y = '{}.01.jpg'.format(Path(filename).stem, i)
            
            flare_x = np.uint8(flare_x)
            flare_y = np.uint8(flare_y)
            
            io.imwrite(osp.join(output_path, 'x_' + split, basename_x), flare_x)
            io.imwrite(osp.join(output_path, 'y_' + split, basename_y), flare_y)
            
    
    


def save_as_arr_x(output_filename, path_fo_files, filenames):
    #depth = io.imread(osp.join(path_fo_files, filenames[0] + '.jpg')).shape[-1]
    depth = 3
    arr = np.empty((len(filenames), FRAGMENT_SIZE, FRAGMENT_SIZE, depth), dtype=np.float32)
    for i, filename in enumerate(filenames):
        img = io.imread(osp.join(path_fo_files, filename + '.jpg'))
        arr[i,:,:,:] = img.reshape(FRAGMENT_SIZE, FRAGMENT_SIZE, depth).astype(np.float32).copy() / 255.
    arr.tofile(output_filename)
    
def save_as_arr_y(output_filename, path_fo_files, filenames):
    #depth = io.imread(osp.join(path_fo_files, filenames[0] + '.jpg')).shape[-1]
    depth = 3
    arr = np.empty((len(filenames), OUTPUT_SIZE, OUTPUT_SIZE, depth), dtype=np.float32)
    for i, filename in enumerate(filenames):
        img = io.imread(osp.join(path_fo_files, filename + '.jpg'))
        arr[i,:,:,:] = img.reshape(OUTPUT_SIZE, OUTPUT_SIZE, depth).astype(np.float32).copy() / 255.
    arr.tofile(output_filename)
    



def process(input_path, flare_type,
            output_path, njobs=DEFAULT_NJOBS):
    
    from joblib import Parallel, delayed

    for split in DATA_SPLITS:
        if not osp.exists(osp.join(input_path, split)):
            raise IOError('No subdirectory `{}` in dir {}'.format(split, input_path))
            
    
    num_files = 0
    unprocessed_filenames = {}
    for split in DATA_SPLITS:
        unprocessed_filenames[split] = sorted(glob.glob(osp.join(input_path, split, '*.jpg')))
        num_files += len(unprocessed_filenames[split])
    

    subdirs = ['x_train', 'x_test', 'x_valid', 'y_valid', 'y_train', 'y_test']
    for subdir in subdirs:
        if not osp.exists(osp.join(output_path, subdir)):
            os.mkdir(osp.join(output_path, subdir))
    
    with Parallel(n_jobs=DEFAULT_NJOBS) as parallel, tqdm.tqdm(total=num_files) as pbar:
        
        def file_producer(filenames, pbar):
            for split in DATA_SPLITS:
                for filename in filenames[split]:
                    pbar.update()
                    yield filename, split

        parallel(delayed(flare_mixup_wrapper)(filename, 
                                              split, output_path) 
                 for filename, split in file_producer(unprocessed_filenames, pbar))
        pbar.close()
    
    for split in DATA_SPLITS:
        preprocessed_filenames = [f.replace('.jpg', '') for f in os.listdir(osp.join(output_path, 'x_' + split))]
        
        pd.DataFrame(preprocessed_filenames).to_csv(osp.join(output_path, split + '_filenames.csv'))     
                
        save_as_arr_y(osp.join(output_path, 'y_' + split + '.bin'), 
                    osp.join(output_path, 'y_' + split), 
                    preprocessed_filenames)
        save_as_arr_x(osp.join(output_path, 'x_' + split + '.bin'), 
                    osp.join(output_path, 'x_' + split),
                    preprocessed_filenames) 
        
            

def main():
    import argparse

    parser = argparse.ArgumentParser('Preprocess dataset for NN training')

    parser.add_argument(
        '-j', '--njobs', type=int, required=False, default=DEFAULT_NJOBS)

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='path to output dir')

    parser.add_argument(
        '-n', '--name', type=str, required=True,
        help='unique name id for the dataset. The resulting name will be '
             '`<total num files>`_preprocessed_<name>')

    parser.add_argument(
        '--flare-type', choices=['cross', 'circle', 'all'], required=False, default='all',
        help='flare type [`cross`, `circle`] if `--sim-flares` is enabled')   

    parser.add_argument(
        'input', type=str,
        help='path to directory with uprocessed jpeg images in format '
             '`<total num files>_unprocessed`, which is'
             'splitted into `train`, `test` and `valid` subdirectories')

    args = parser.parse_args()

    if not osp.exists(args.input):
        raise IOError('No such input path `{}`'.format(args.input))

    if not osp.exists(args.output):
        raise IOError('No such output dir `{}`'.format(args.output))

    if len(args.input.split('_')) != 2 or \
       args.input.split('_')[1] != 'unprocessed':
        raise IOError('Wrong unprocessed dataset name, see help.')

    
    dataset_dir_name = '{}_preprocessed_{}'.format(
        osp.basename(args.input.split('_')[0]), args.name
    )    

    if not osp.exists(osp.join(args.output, dataset_dir_name)):
        os.mkdir(osp.join(args.output, dataset_dir_name))

    process(args.input, 'all', 
            osp.join(args.output, dataset_dir_name), 
            args.njobs)


if __name__ == '__main__':
    main()
