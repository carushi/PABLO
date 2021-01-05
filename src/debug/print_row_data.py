import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

png_flag = False
file_list = ['GD.nii.gz', '2wGD.nii.gz']

for i, file in enumerate(file_list):
    if not os.path.exists(file):
        continue
    p = nib.load(file.data).dataobj
    name = str(i)
    print(name, p.shape)
    if png_flag:
        for j in range(p.shape[2]):
            print(i)
            plt.imshow(p[:,:,j])
            plt.savefig(name+'_'+str(j)+'.png')
            plt.close()
            plt.clf()
    else:
        images = []
        for i in range(p.shape[2]):
            im = Image.open(name+'_'+str(i)+'.png')
            images.append(im)
        images[0].save(name+'.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
