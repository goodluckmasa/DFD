import os
from options.test_options import TestOptions
from models import create_model
# from util.visualizer import save_crop
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from data.image_folder import make_dataset
import sys
from util import util
from DFLIMG.DFLJPG import DFLJPG
import shutil

OUT_RES = 640

def get_part_location(lmrks):
    Landmarks = []
    
    Landmarks = np.array(lmrks)
    
    Map_LE = list(np.hstack((range(17,22), range(36,42))))
    Map_RE = list(np.hstack((range(22,27), range(42,48))))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))
    try:
        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except:
        return 0
    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)
    
if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.gpu_ids = [0] # gpu id. if use cpu, set opt.gpu_ids = []
    WorkingDir = opt.working_dir
    tmp_dir = opt.tmp_dir
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    #OutputPath = opt.results_dir
    InputPath = os.path.join(WorkingDir,'aligned')
    OutputPath = os.path.join(WorkingDir,'aligned_dfdn')
    os.makedirs(OutputPath, exist_ok=True)
    jpeg_quality = opt.jpeg_quality
    model = create_model(opt)
    model.setup(opt)
    
    counter = 0
    # for i, ImgPath in enumerate(pathex.get_image_paths(InputPath, return_Path_class=True)):
    ImgPaths = make_dataset(InputPath)
    for i, ImgPath in enumerate(ImgPaths):
        counter += 1
        ProgressStr = '[{}/{}]'.format(counter, len(ImgPaths))
        ImgName = os.path.split(ImgPath)[-1]
        if os.path.isfile(os.path.join(OutputPath, ImgName)):
            print('{} Skipping {}'.format(ProgressStr, ImgName))
            continue
        print('{} Restoring {}'.format(ProgressStr, ImgName))
        torch.cuda.empty_cache()
        
        if tmp_dir:
            tmp_path = os.path.join(tmp_dir,ImgName)
            shutil.copyfile(ImgPath, tmp_path)
            work_path = tmp_path
        else:
            work_path = ImgPath
        
        InputDflImg = DFLJPG.load(work_path)
        if not InputDflImg or not InputDflImg.has_data():
            print('\t################ No landmarks in file {}'.format(ImgName))
            continue
        Landmarks = InputDflImg.get_landmarks()
        InputData = InputDflImg.get_dict()
        
        # scale landmarks and xseg polys to output image size
        scale_factor = OUT_RES / InputDflImg.get_shape()[0]
        # print('Scale factor: {}'.format(scale_factor))
        Landmarks = Landmarks * scale_factor
        if InputDflImg.has_seg_ie_polys():
            xseg_polys = InputDflImg.get_seg_ie_polys()
            for poly in xseg_polys:
                poly.set_points(poly.get_pts() * scale_factor)
        
        Part_locations = get_part_location(Landmarks)
        A = Image.open(work_path).convert('RGB')
        
        if Part_locations == 0:
            print('\t################ Error in landmarks, continue...')
            continue
        C = A
        A = A.resize((OUT_RES, OUT_RES), Image.BICUBIC)
        A = transforms.ToTensor()(A) 
        C = transforms.ToTensor()(C)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)
        
        data = {'A':A.unsqueeze(0), 'C':C.unsqueeze(0), 'A_paths': work_path,'Part_locations': Part_locations}

        model.set_input(data)
        try:
            model.test()
            visuals = model.get_current_visuals()
            
            im_data = visuals['fake_A']
            im = util.tensor2im(im_data)
            image_pil = Image.fromarray(im)
            if tmp_dir:
                save_path = os.path.join(tmp_dir,'tmp'+ImgName)
            else:
                save_path = os.path.join(OutputPath,ImgName)
            image_pil.save(save_path, quality=jpeg_quality)
            
        except Exception as e:
            print('\t################ Error enhancing {}'.format(str(e)))
            os.remove(work_path)
            continue
        
        OutputDflImg = DFLJPG.load(save_path)
        OutputDflImg.set_dict(InputData)
        OutputDflImg.set_landmarks(Landmarks)
        if InputDflImg.has_seg_ie_polys():
            OutputDflImg.set_seg_ie_polys(xseg_polys)
        OutputDflImg.save()
        if tmp_dir:
            shutil.copyfile(save_path, os.path.join(OutputPath,ImgName))
            os.remove(work_path)
            os.remove(save_path)

    print('\nAll results are saved in {} \n'.format(OutputPath))
    
