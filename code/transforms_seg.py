import torch 
import numpy as np
from skimage import transform

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, pred = sample['image'], sample['pred']

        # print(image.shape)
        # print(type(image))
        # print(gt.shape)
        # print(type(gt))
        #print(t,h,w)
        
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size
        # new_t = 128
        # new_h, new_w = int(new_h), int(new_w)
        #print(image.shape)
        #img = transform.resize(image, (new_t,new_h, new_w ))
        #print("Despues",img.shape)
        #gt = transform.resize(gt, (new_t, new_h, new_w))
        
        return {'image': image, 'pred': pred}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        prediction=sample['pred']
        #image = image.transpose((0, 3, 1, 2))
        #gt = gt.transpose((0, 3, 1, 2))
        #image = np.expand_dims(image, axis=0)
        #gt = np.expand_dims(gt, axis=0)
        #image = image[np.newaxis, :]
        #gt = gt[np.newaxis, :]
        #print(image.shape)
        #img_t = torch.from_numpy(image)
        #print(gt.shape)
        #image = np.tile(image, (2,1,1,1,1))
        #gt = np.tile(gt, (2,1,1,1,1))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_var_torch = torch.from_numpy(image).to(torch.float) / 255.0
        pred_tensor = torch.from_numpy(prediction).to(torch.float) / 255.0
        
        # print(torch.min(image_var_torch), torch.max(image_var_torch))
        # print(torch.min(gt_var_torch), torch.max(gt_var_torch))
        return {'image': image_var_torch, 'pred': pred_tensor}
