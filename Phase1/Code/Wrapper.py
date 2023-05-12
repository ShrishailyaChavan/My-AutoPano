#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import copy
import argparse
import numpy as np
import cv2


class ANMS:
    def __init__(self, N):
        self.N_points = N

    def ANMS(self, total_corners):
        best_corners = list()
        values = list()


        for i in range(len(total_corners)):
            if i == 0:
                best_corners.append([total_corners[i][0][0], total_corners[i][0][1]])
                continue
            Euclidean_Distance = total_corners[:i] - total_corners[i][0]     # since all above it have strong corner response value
            Euclidean_Distance = Euclidean_Distance**2
            Euclidean_Distance = np.sum(Euclidean_Distance, axis=2)
            minimum_Euclidean_Distance= min(Euclidean_Distance)
            index = np.where(minimum_Euclidean_Distance==Euclidean_Distance)
            if [total_corners[index[0][0]][0][0], total_corners[index[0][0]][0][1]] in best_corners:
                continue
            best_corners.append([int(total_corners[index[0][0]][0][0]), int(total_corners[index[0][0]][0][1])])
            values.append(minimum_Euclidean_Distance[0])
        values = np.array(values)
        index = np.argsort(values)
        best_corners = np.array(best_corners)
        best_corners = best_corners[index]
        return best_corners[:self.N_points]

class Feature_Descriptor:
    def __init__(self, Patch_Size=40):
        self.Patch_Size = Patch_Size

    def Feature_Descriptors(self, image, corner_points, variable, SAVE_IMAFE_AS):
        
        Total_features = list()
        total_corners_of_patch = 1
        for item in corner_points:
            bottom_y, bottom_x = (0,0)
            bottomy = item[1]-int(20)
            if bottomy < 0:
                bottomy = item[1] + abs(bottomy) - int(20)
                bottom_y = abs(item[1]-int(20))  

            uppery = item[1]+int(20) + bottom_y
            if uppery > image.shape[0]:
                diff = uppery - image.shape[0]
                uppery = item[1] - diff + int(20)
                bottomy = bottomy - diff

            lowerx = item[0]-int(20)
            if lowerx < 0:
                lowerx = item[0] + abs(lowerx) - int(20)
                bottom_x = abs(item[0]-int(20))   
            upperx = item[0]+int(20)  + bottom_x
            if upperx > image.shape[1]:
                diff = upperx - image.shape[1]
                upperx = item[0] - diff + int(20)
                lowerx = lowerx - diff

            #Applying the patch around the points
            img_patch = image[bottomy:uppery, lowerx:upperx]
            #applyting gaussian blur 
            gaussian_blur = cv2.GaussianBlur(img_patch, (3,3),0)
            feat_desc = cv2.resize(gaussian_blur, (8,8), interpolation = cv2.INTER_CUBIC)
            #resizing the blurred output to 8x8 using cv2.resize
            if total_corners_of_patch < 5 and SAVE_IMAFE_AS:
                cv2.imwrite("results/FeatureDescriptor/FD%s%s.png"%(variable, total_corners_of_patch), feat_desc)
            feat_desc = np.reshape(feat_desc, (feat_desc.shape[0]*feat_desc.shape[1]))
            mean = np.sum(feat_desc)/feat_desc.shape[0]
            std = ((1/feat_desc.shape[0])*(np.sum((feat_desc-mean)**2)))**(1/2)
            #applying the standardization 
            feat_desc = (feat_desc - (mean))/std
            Total_features.append(feat_desc)
            total_corners_of_patch = total_corners_of_patch + 1
        return Total_features

    #def FeatureDescriptor(gray_image):


    #Get the feature points 
        #points = cv2.goodFeaturesToTrack(gray_image, 1000, 0.01, 10)

    #Create patches of size 41x41 around each feature point
        #patches = []

       # for point in points:
        #Get the coordinates of the feature point
      #      x, y = point.ravel()
      #      #create the patch 
      #      patch = gray_image[x-20:x+20, y-20:y+20]
      #      patches.append(patch)

      #      patches = cv2.GaussianBlur(patches, (3,3), 0)

       #      patches = cv2.resize(patches, (8,8), interpolation = cv2.INTER_CUBIC)

       #     feature = patches.reshape(64,1)
    

       #     feature = (feature-feature.mean()) / np.std(feature)

       # return feature

    def distance(descriptor_1, descriptor_2):
        s = 0
        for i in range(len(descriptor_1)):
            s = s + (descriptor_1[i] - descriptor_2[i])**2

        return s



    def matchFeatures(self, descriptor_1, descriptor_2):
        matched_pairs = list()
        descriptor_1 = np.array(descriptor_1, dtype=np.int64)
        descriptor_2 = np.array(descriptor_2, dtype=np.int64)
        for i in range(len(descriptor_1)):
            SSD = descriptor_2 - np.reshape(descriptor_1[i], (1,descriptor_1.shape[1]))
            SSD = SSD**2
            SSD = np.sum(SSD, axis = 1)

            minSSD = np.min(SSD)
            if minSSD > 15:
                continue
            index = np.where(SSD==minSSD)
            matched_pairs.append([i,index[0][0]])

    
        threshold_ratio = len(matched_pairs)/len(descriptor_1)
        return matched_pairs




    # def getfeature_matches(descriptor_1, descriptor_2, threshold_ratio= 0.75):

  #  ratio = minimum_distance/second_minimum_distance
  #  points_1 = []
  #  points_2 = []


  #  points_distance = []
    

  #  for d1 in range(len(descriptor_1)):
  #      minimum_distance = float('inf')
  #      second_minimum_distance = float('inf')
   #     second_point = None

   #     for d2 in range(len(descriptor_2)):
   #         s = distance(d1, d2)
   #         if s < minimum_distance:
              #  second_minimum_distance = minimum_distance
   #             minimum_distance = s

   #         elif s < second_minimum_distance:
   #             second_minimum_distance = s

   #             second_point = d2

  #      if ratio < threshold_ratio:

  #          points_distance.append(minimum_distance)
   #         points_1.append(d1)
  #          points_2.append(d2)
    
   # points_1 = np.array(points_1)
 #   points_2 = np.array(points_2)
  #  points_distance = np.array(points_distance)

   # return points_1, points_distance, points_2


    def draw_features(self, feature_points_1, img1, feature_points_2, img2, matched_pairs, variable, SAVE_IMAFE_AS):
        MAXIMUM_HEIGHT_OF_IMAGE = np.max([img1.shape[0], img2.shape[0]])
        img_1 = np.zeros((MAXIMUM_HEIGHT_OF_IMAGE,img1.shape[1],img1.shape[2]),dtype=np.uint8)
        img_2 = np.zeros((MAXIMUM_HEIGHT_OF_IMAGE,img2.shape[1],img1.shape[2]),dtype=np.uint8)
        img_1[:img1.shape[0],:img1.shape[1]] = img1
        img_2[:img2.shape[0],:img2.shape[1]] = img2
        joint_imgs = np.concatenate((img_1, img_2), axis=1)
        for i in range(len(matched_pairs)):
            start = (feature_points_1[matched_pairs[i][0]][0], feature_points_1[matched_pairs[i][0]][1])
            end = (feature_points_2[matched_pairs[i][1]][0]+img1.shape[1], feature_points_2[matched_pairs[i][1]][1])
            cv2.line(joint_imgs, start, end, (0,0,255), 1)
        cv2.imshow("Feature_Matching",joint_imgs)
        if SAVE_IMAFE_AS:
            cv2.imwrite("results/MatchingOutput/matching%s.png"%(variable), joint_imgs)
        cv2.waitKey(1)
        return

    #def draw_features(image_1, image_2, points_1, points_2, points_distance):
   # matches = []
   # for i, distnace_match in enumerate(points_distance):
   #     matches.append(cv2.DMatch(i,i, points_distance[i]))
    
   # ret = np.array([])
   # Visualization = cv2.draw_featureses(img1=image_1,
    #    keypoints1=points_1,
   #     img2=image_2,
    #    keypoints2=points_2,
    #    matches1to2=matches,matchesThickness=1, outImg = ret)


   # return Visualization

class RANSAC:

    def SSD(self, pts1, pts2):
        #Assume pts1 and pts2 are lists of matching points of the same size
        SSD = np.square(pts1 - pts2)
        SSD = np.sum(SSD, axis=1)
        inlier = np.where(SSD < 2)
        return inlier

        #ssd = 0
        #for p1, p2 in zip(pts1, pts2):
        #    ssd = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        #    ssd = np.sum(ssd)
        #    inlier = np.where(ssd < 2)
        #    return inlier


    def Random_Sampling_Consensus(self, points1, points2, matching_pairs, n_iterations = 15000):
        #converting matching_pairs to array/matrix form
        matching_pairs = np.array(matching_pairs)
        value = 0
        best_inliers = None
        #index = np.arange(matching_pairs.shape[0])
        left_side_points = points1[matching_pairs[:,0]]
        right_side_points = points2[matching_pairs[:,1]]
    
        
        while True:
            value = 0
            while n_iterations > value:
                #Randomly selecting four points from the set
                rand_pair_dxs = np.random.choice(len(matching_pairs), size=4, replace=False)
                #calculating Homography using the randomly selected points
                first_point = left_side_points[rand_pair_dxs]
                second_point = right_side_points[rand_pair_dxs]

                #here we apply get perspective transform where we get an output of 3x3 matrix 

                H = cv2.getPerspectiveTransform(np.float32(first_point), np.float32(second_point))
                points_before = np.concatenate((left_side_points, np.ones((left_side_points.shape[0],1))), axis = 1)
                points_after = np.dot(H, points_before.T)
                points_after[-1,:] = points_after[-1,:]+0.0001
                points_after = points_after/(points_after[-1,:])
                points_after = points_after.T

                #Finding the inliners between right side points and points with different perspective after applying Perspective Transform

                inliers = self.SSD(right_side_points, points_after[:,:2])
                number_of_inliers = len(inliers[0])
                if best_inliers == None or best_inliers < number_of_inliers:
                    best_inliers = number_of_inliers
    
                    best_points1 = left_side_points[inliers]
                    best_points2 = right_side_points[inliers]
                    H_best = H
                value = value + 1

           
            if best_inliers > 6:
                break
        return best_points1, best_points2, H_best, best_inliers

    def draw_features(self, feature_points_1, img1, feature_points_2, img2, variable, SAVE_IMAFE_AS):
        max_height = np.max([img1.shape[0], img2.shape[0]])
        img_1 = np.zeros((max_height,img1.shape[1],img1.shape[2]),dtype=np.uint8)
        img_2 = np.zeros((max_height,img2.shape[1],img1.shape[2]),dtype=np.uint8)
        img_1[:img1.shape[0],:img1.shape[1]] = img1
        img_2[:img2.shape[0],:img2.shape[1]] = img2
        joint_imgs = np.concatenate((img_1, img_2), axis=1)

        for i in range(len(feature_points_1)):
            beginning_point = (feature_points_1[i,0], feature_points_1[i,1])
            end_point = (feature_points_2[i,0]+img1.shape[1], feature_points_2[i,1])
            cv2.line(joint_imgs, beginning_point, end_point, (0,0,255), 1)
        cv2.imshow("RANSAC",joint_imgs)
        if SAVE_IMAFE_AS:
            cv2.imwrite("results/RansacOutput/RANSACmatching%s.png"%(variable), joint_imgs)
        cv2.waitKey(1)
        return

    def Find_Homography(self, src_points, dst_points):
        X = np.zeros((2*len(src_points),9))
        i = 0
        for x in range(len(X)):
            if x%2 == 0:
                X[x,:] = [src_points[i][0], src_points[i][1], 1, 0, 0, 0, -(dst_points[i][0] * src_points[i][0]), -(dst_points[i][0] * src_points[i][1]), -dst_points[i][0]]
            else:
                X[x,:] = [0, 0, 0, src_points[i][0], src_points[i][1], 1, -(dst_points[i][1] * src_points[i][0]), -(dst_points[i][1] * src_points[i][1]), -dst_points[i][1]]
                i += 1

        #here we have U and V as unitary and S as a rectangular diagonal matrix containing the singular values of the matrix
        U, S, V = np.linalg.svd(X)
        Vt = V.T
        h = Vt[:,8]/Vt[8][8]
        H = np.reshape(h, (3,3))
        return H

class Blending_images:
    def __init__(self,number_of_images_to_blend):
        self.num = number_of_images_to_blend

    def New_Dimension(self, other_img, H):
        new_point = np.array([[0,0,1],[0,other_img.shape[0],1],[other_img.shape[1],other_img.shape[0],1],[other_img.shape[1],0,1]])
        #calculating the border value or end value
        end_value = np.dot(H, new_point.T)
        end_value = end_value/end_value[-1]
       #calculating the minimum and maximum values of row 
        row_minimum = np.min(end_value[1,:])
        row_maximum = np.max(end_value[1,:])
        #calculating the minimum and maximum values of column
        column_minimum = np.min(end_value[0,:])
        column_maximum = np.max(end_value[0,:])
    
        if column_minimum < 0:
            new_width_of_image = round(column_maximum - column_minimum)
        else:
    
            new_width_of_image = round(column_maximum - column_minimum)

        if row_minimum < 0:
            new_height_of_image = round(row_maximum - row_minimum)
        else:
            # new_height = round(row_max)
            new_height_of_image = round(row_maximum - row_minimum)
        calculating_shift = np.array([[1,0,-column_minimum],[0,1,-row_minimum],[0,0,1]])
        H = np.dot(calculating_shift, H)
        return new_height_of_image, new_width_of_image, H

    def Calculate_Translation(self, points1, points2, H):
        #converting both points to array/matrix format
        points1 = np.array(points1)

        points2 = np.array(points2)
        #suppose here we take four points from both sides each
        input = np.ones((points2.shape[0],1))
        #here we get 4x1 matrix
        points = np.concatenate((points2,input), axis=1)
        #further we get 4x3 matrix
        point = points.T
        #after applying transformation we get 3x4 matrix
        point_after_transformation = np.dot(H,point)
        # here dimensions of transformed point are 3x4
        point_after_transformation = point_after_transformation/point_after_transformation[-1]
        #outcome of the matrix after this operation is 3x4 matrix

        points1 = points1.T
        #here after transformation we get matrix shape as 2x4

        translations = points1 - point_after_transformation[:2]
        translations = translations.T
        #here after transformation we get 4x2 matrix

        sum_of_translations = np.sum(translations, axis=0)
        mean_of_translation = sum_of_translations/translations.shape[0]
        Final_translation = np.array([[mean_of_translation[0]],[mean_of_translation[1]]])
        #here finally we get shape of 2x1 for translation
        return Final_translation, point_after_transformation

    #stitching first part to second part
    def stitch_the_images(self, stitch_first_part, to_second_part, point_after_transformation):
        later_shape = to_second_part.shape
        previous_shape = stitch_first_part.shape
        #converting both later_shape and previous_shape to array/matrix form
        shape_of_img = np.array([later_shape,previous_shape])
        create_panorama = np.zeros((np.max(shape_of_img[:,0]), np.max(shape_of_img[:,1]),3), dtype=np.uint8)
        create_panorama[:later_shape[0],:later_shape[1]] = to_second_part
        index = np.where(stitch_first_part>0)
        create_panorama[index[:2]] = stitch_first_part[index[:2]]
        return create_panorama


    def warp_and_stitch_points(self, img1, img1_points, img2, img2_points, H):
        #finding the multiplicative inverse of the matrix
        H = np.linalg.inv(H)
        new_height_of_image, new_width_of_image, H = self.New_Dimension(img2, H)
        stitch_first_part = cv2.warpPerspective(img2, H, (new_width_of_image,new_height_of_image))

        translation, point_after_transformation = self.Calculate_Translation(img1_points, img2_points, H)

        if translation[0,0] < 0 and translation[1,0] < 0:
            Calculating_translation = np.float32([[1,0,np.absolute(round(translation[0,0]))],[0,1,np.absolute(round(translation[1,0]))]])
            new_shape = (np.absolute(round(translation[0,0]))+img1.shape[1], np.absolute(round(translation[1,0]))+img1.shape[0])
            img1 = cv2.warpAffine(img1, Calculating_translation, new_shape)
            # panorama = self.stitch(stitch_this, img, point_after_transformation)
            final_panorama = self.stitch_the_images(img1, stitch_first_part, point_after_transformation)
        elif translation[0,0] < 0 and translation[1,0] > 0:
            Calculating_translation = np.float32([[1,0,np.absolute(round(translation[0,0]))],[0,1,0]])
            new_shape = (np.absolute(round(translation[0,0]))+img1.shape[1], img1.shape[0])
            img1 = cv2.warpAffine(img1, Calculating_translation, new_shape)

            Calculating_translation = np.float32([[1,0,0],[0,1,np.absolute(round(translation[1,0]))]])
            new_shape = (stitch_first_part.shape[1], np.absolute(round(translation[1,0]))+stitch_first_part.shape[0])
            stitch_first_part = cv2.warpAffine(stitch_first_part, Calculating_translation, new_shape)
            final_panorama = self.stitch_the_images(img1, stitch_first_part, point_after_transformation)
        elif translation[0,0] > 0 and translation[1,0] < 0:
            Calculating_translation = np.float32([[1,0,np.absolute(round(translation[0,0]))],[0,1,0]])
            new_shape = (np.absolute(round(translation[0,0]))+stitch_first_part.shape[1], stitch_first_part.shape[0])
            stitch_first_part = cv2.warpAffine(stitch_first_part, Calculating_translation, new_shape)

            Calculating_translation = np.float32([[1,0,0],[0,1,np.absolute(round(translation[1,0]))]])
            new_shape = (img1.shape[1], np.absolute(round(translation[1,0]))+img1.shape[0])
            img1 = cv2.warpAffine(img1, Calculating_translation, new_shape)
            final_panorama = self.stitch_the_images(img1, stitch_first_part, point_after_transformation)
        else:
            Calculating_translation = np.float32([[1,0,round(translation[0,0])],[0,1,round(translation[1,0])]])
            new_shape = (np.absolute(round(translation[0,0]))+stitch_first_part.shape[1], abs(round(translation[1,0]))+stitch_first_part.shape[0])
            stitch_first_part = cv2.warpAffine(stitch_first_part, Calculating_translation, new_shape)
            final_panorama = self.stitch_the_images(img1, stitch_first_part, point_after_transformation)

        cv2.imshow("panorama", final_panorama)
        cv2.waitKey(1)

        return final_panorama




def main(): 
#Accept arguments from command line

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--path', type=str, default='Train/Set1',
                        help='Source Image Path for Training')
    Parser.add_argument('--save', type=bool, default=True,
                        help='Flag to save the image')

    #Process the path values

    Args = Parser.parse_args()
    path = Args.path
    variable = path.split('/')
    variable = variable[1]
    SAVE_IMAFE_AS = Args.save
    """
    Read a set of images for Panorama stitching
    """

    Total_Input_Images = 0

    x = 1

    Images_as_input = list()
    while True:
       #Read all the source images from the example Data/Train/Set1
        img = cv2.imread("../Data/%s/%s.jpg"%(path,x))
        x = x + 1
        try:
            image_shape = img.shape
            Images_as_input.append(img)
            Total_Input_Images += 1
        except:
            break
    
    #keeping the following condition, if we are provided with more then 4 images to stitch
    div_factor = Total_Input_Images
    if Total_Input_Images > 5:
        div_factor = Total_Input_Images - 5

    #changing the size of image
    
    proportional_ratio = Images_as_input[0].shape[0]/Images_as_input[0].shape[1]
    w_scaling = 1280/(div_factor*Images_as_input[0].shape[1])
    w = int(Images_as_input[0].shape[1]*w_scaling)
    h = int(proportional_ratio*w)

    #resizing the image size for the final outcomes
    images = []
    for i in Images_as_input:
        resize_image = cv2.resize(i, (w, h))
        images.append(resize_image)
    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    def create_Panorama(Total_Input_Images):
        gray_image = []
        for i in Total_Input_Images:
            gray_img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            gray_image.append(gray_img)

        print("Begin corner detection using SHi-Tomasi method.")

        corners = []
        for i in gray_image:
            finding_total_corners = cv2.goodFeaturesToTrack(i, 10000, 0.001, 6)
            
            corners.append(finding_total_corners)

        corners = [np.intp(i) for i in corners]

        Total_corners = copy.deepcopy(Total_Input_Images)
        for i in range(len(Total_corners)):
            img = Total_corners[i]
            Circle_corners = corners[i]
            for i in Circle_corners:
                x, y = i.ravel()
                cv2.circle(img, (x,y), 3, (0, 0,255), -1)

        if SAVE_IMAFE_AS:
            for i in range(len(Total_corners)):
                cv2.imwrite("results/Corners/corners%s%s.png"%(variable,i+1), Total_corners[i])

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        images_anms = copy.deepcopy(Total_Input_Images)
        apply_anms = ANMS(1100)

        N_best_points = list()
        for i in range(len(images_anms)):
            N_best_points.append(apply_anms.ANMS(corners[i]))

        for i in range(len(N_best_points)):
            for j in N_best_points[i]:
                cv2.circle(images_anms[i], (j[0],j[1]), 3, (0, 0, 255), -1)


        cv2.imshow("ANMS",images_anms[0])
        cv2.imshow("ANMS__",images_anms[1])
        cv2.waitKey(1)
        if SAVE_IMAFE_AS:
            for i in range(len(images_anms)):
                cv2.imwrite("results/ANMS/anms%s%s.png"%(variable,i+1), images_anms[i])

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        total_descriptors = list()
        features = Feature_Descriptor()
        for i in range(len(gray_image)):
            total_descriptors.append(features.Feature_Descriptors(gray_image[i], N_best_points[i], variable, SAVE_IMAFE_AS))

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        match_pairs = list()
        for i in range(len(total_descriptors)-1):
            match_pairs.append(features.matchFeatures(total_descriptors[i], total_descriptors[i+1]))

        for i in range(len(match_pairs)):
            features.draw_features(N_best_points[i], Total_Input_Images[i], N_best_points[i+1], Total_Input_Images[i+1], match_pairs[i], variable, SAVE_IMAFE_AS)

        """
        Refine: RANSAC, Estimate Homography
        """
        ransac_refine = RANSAC()
        best_points_1, best_points_2, best_Homography, inliers = ransac_refine.Random_Sampling_Consensus(N_best_points[0], N_best_points[1], match_pairs[0])
        autopano = Total_Input_Images[0]
        if inliers >= 6:
            ransac_refine.draw_features(best_points_1, Total_Input_Images[0], best_points_2, Total_Input_Images[1], variable, SAVE_IMAFE_AS)
            H = ransac_refine.Find_Homography(best_points_1, best_points_2)
        
            """
            Image Warping + Blending
            Save Panorama output as mypano.png
            """
            Blending_of_images = Blending_images(1)
            autopano = Blending_of_images.warp_and_stitch_points(Total_Input_Images[0], best_points_1, Total_Input_Images[1], best_points_2, H)
        
        return autopano

      # Below is the logic to start from center and stitch either side of processed images
    if Total_Input_Images%2 == 0:
        _index_value = int(len(images)/2)-1
    else:
        _index_value = int((len(images)/2))
    img_ = [images[_index_value], images[_index_value-1]]

    count = 1
    for i in range(1,len(images)):
        autopano = create_Panorama(img_)
        if SAVE_IMAFE_AS:
            cv2.imwrite("results/Panoramas/mypano%s.png"%(variable), autopano)
        
        if i%2 == 0:
            img_ = [autopano, images[_index_value - count]]
        else:
            img_ = [autopano, images[_index_value + count]]
            count = count + 1
    autopano = cv2.resize(autopano, (800,800))
    cv2.imshow("Final Panoramic Image",autopano)
    if SAVE_IMAFE_AS:
        cv2.imwrite("results/Panoramas/mypano%s.png"%(variable), autopano)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()