#!/usr/bin/env python
import os
import cv2
import numpy as np
import torch
from Network.Network import HomographySupModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_path = '/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Checkpoints/ckpt29.pt' 

def stitch_images(images_folder, model, device='cpu'):
    # Get the list of images in the folder    
    checkPt = torch.load(weights_path)
    model.load_state_dict(checkPt)
    # model.eval()
    images_list = sorted(os.listdir(images_folder))
    images = []
    for image in images_list:
        # Load the images and convert to PyTorch tensor
        img = cv2.imread(os.path.join(images_folder, image))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        # img = torch.tensor(img, dtype=torch.float32)
        # img = img.permute(2, 0, 1).unsqueeze(0)
        images.append(img)
    
    # Initialize the first image as the result image
    result = images[0]
    transformed_imgs = [result]
    print(result.shape)
    for i in range(1, len(images)):
    # for i in range(1, 7):
        # Get the homography matrix from the model
        img1 = result
        # print(img1.shape)        
        img2 = images[i]
        # print(img2.shape)
        stack = torch.from_numpy(np.swapaxes(np.dstack((img1, img2)), 2, 0)).unsqueeze(0)
        # print(stack.shape)
        homography = model(stack).cpu().detach().numpy()
        print(homography.shape)
        homography = np.array([[homography[0][0], homography[0][1], homography[0][2]], [homography[0][3], homography[0][4], homography[0][5]], \
            [homography[0][6], homography[0][7], 1]])
        # print(homography.shape)
        # print(images[i])
        # Apply the homography matrix to align the next image
        img_warped = cv2.warpPerspective(img2, homography, (result.shape[1]+img2.shape[1], result.shape[0]))
        img_warped = cv2.resize(img_warped, (128,128))
        result = img_warped
        # result = np.max(result, img)
        # cv2.imwrite('/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase1/Data/Test/TestSet2/stitched'+str(i)+'.jpg', result)
        transformed_imgs.append(result)

        # img2 = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        # print(img2.shape)
    #     # result = torch.max(result, img2)
    
    # print(transformed_imgs)
    stitched_image = np.concatenate(transformed_imgs, axis=1)
    
    # return result.permute(0, 2, 3, 1).squeeze().numpy()
    return stitched_image

def main():
    img_folder = "/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase1/Data/Test/TestSet4/"
    # weights_path = '/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Checkpoints/ckpt20.pt'    
    model = HomographySupModel().to(device)
    # model.load_state_dict(torch.load(weights_path))
    checkPt = torch.load(weights_path)
    model.load_state_dict(checkPt)

    result = stitch_images(img_folder, model, device=device)

    cv2.imwrite('/home/jc-merlab/RBE549-Computer_Vision/schatterjee_p1/Phase1/Data/Test/TestSet4/stitched.jpg', result)

if __name__ == '__main__':
    main()



    