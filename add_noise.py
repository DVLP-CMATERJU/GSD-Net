import os
import skimage.io as io
import numpy as np
import shutil
def add_noise(data_path,sigma,out_dir):
    splits = ['train','validation','test']
    for split in splits : 
        images = os.listdir(data_path+'/'+split+'/images')
        for image in images:
            image_path = data_path+'/'+split+'/images/'+image
            label_path = data_path+'/'+split+'/labels/'+image
            img_data = io.imread(image_path, as_gray=True)
            img_data = img_data/255.0
            noise = np.random.normal(scale = sigma, size = img_data.shape)
            img_data = img_data+noise
            img_data = (img_data-img_data.min())/(img_data.max()-img_data.min())
            img_data = (img_data*255).astype(np.uint8)
            if not os.path.exists(out_dir+'/'+split+'/images/'):
                os.makedirs(out_dir+'/'+split+'/images/')
            io.imsave(out_dir+'/'+split+'/images/'+image,img_data)
            if not os.path.exists(out_dir+'/'+split+'/labels/'):
                os.makedirs(out_dir+'/'+split+'/labels/')
            shutil.copyfile(label_path, out_dir+'/'+split+'/labels/'+image)

datasets=['circuits_data_split', 'floor_plan_data_split']
sigmas = [0.01,0.1,0.25,0.5,0.75]
for d in datasets:
    for s in sigmas:
        outpath = d.split('_')[0]+'_'+str(s).split('.')[-1]
        add_noise(d,s,outpath)