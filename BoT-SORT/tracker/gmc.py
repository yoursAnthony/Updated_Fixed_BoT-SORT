import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


class GMC:
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))

        self.last_quality = 1.0
        self.prev_H = np.eye(2, 3, dtype=float)
        self.ema = 0.2  # сглаживание матрицы H

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=2000, qualityLevel=0.001, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)
            # self.gmc_file = open('GMC_results.txt', 'w')

        elif self.method == 'file' or self.method == 'files':
            seqName = verbose[0]
            ablation = verbose[1]
            if ablation:
                filePath = r'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = r'tracker/GMC_files/MOTChallenge'

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]

            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')

            if self.gmcFile is None:
                raise ValueError("Error: Unable to open GMC file in directory:" + filePath)
        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown CMC method:" + method)

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None):
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'none':
            return np.eye(2, 3)
        else:
            return np.eye(2, 3)

    def applyEcc(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=float)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        # (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            print('Warning: find transform failed. Set warp as identity')

        return H

    # def applyFeaures(self, raw_frame, detections=None):

    #     # Initialize
    #     height, width, _ = raw_frame.shape
    #     frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    #     H = np.eye(2, 3)

    #     # Downscale image (TODO: consider using pyramids)
    #     if self.downscale > 1.0:
    #         # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
    #         frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
    #         width = width // self.downscale
    #         height = height // self.downscale

    #     # find the keypoints
    #     mask = np.zeros_like(frame)
    #     # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
    #     mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
    #     if detections is not None:
    #         for det in detections:
    #             tlbr = (det[:4] / self.downscale).astype(np.int_)
    #             mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

    #     keypoints = self.detector.detect(frame, mask)

    #     # compute the descriptors
    #     keypoints, descriptors = self.extractor.compute(frame, keypoints)

    #     # Handle first frame
    #     if not self.initializedFirstFrame:
    #         # Initialize data
    #         self.prevFrame = frame.copy()
    #         self.prevKeyPoints = copy.copy(keypoints)
    #         self.prevDescriptors = copy.copy(descriptors)

    #         # Initialization done
    #         self.initializedFirstFrame = True

    #         return H

    #     # Match descriptors.
    #     knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

    #     # Filtered matches based on smallest spatial distance
    #     matches = []
    #     spatialDistances = []

    #     maxSpatialDistance = 0.25 * np.array([width, height])

    #     # Handle empty matches case
    #     if len(knnMatches) == 0:
    #         # Store to next iteration
    #         self.prevFrame = frame.copy()
    #         self.prevKeyPoints = copy.copy(keypoints)
    #         self.prevDescriptors = copy.copy(descriptors)

    #         return H

    #     for m, n in knnMatches:
    #         if m.distance < 0.9 * n.distance:
    #             prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
    #             currKeyPointLocation = keypoints[m.trainIdx].pt

    #             spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
    #                                prevKeyPointLocation[1] - currKeyPointLocation[1])

    #             if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
    #                     (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
    #                 spatialDistances.append(spatialDistance)
    #                 matches.append(m)

    #     meanSpatialDistances = np.mean(spatialDistances, 0)
    #     stdSpatialDistances = np.std(spatialDistances, 0)

    #     inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

    #     goodMatches = []
    #     prevPoints = []
    #     currPoints = []
    #     for i in range(len(matches)):
    #         if inliesrs[i, 0] and inliesrs[i, 1]:
    #             goodMatches.append(matches[i])
    #             prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
    #             currPoints.append(keypoints[matches[i].trainIdx].pt)

    #     prevPoints = np.array(prevPoints)
    #     currPoints = np.array(currPoints)

    #     # Draw the keypoint matches on the output image
    #     if 0:
    #         matches_img = np.hstack((self.prevFrame, frame))
    #         matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
    #         W = np.size(self.prevFrame, 1)
    #         for m in goodMatches:
    #             prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
    #             curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
    #             curr_pt[0] += W
    #             color = np.random.randint(0, 255, (3,))
    #             color = (int(color[0]), int(color[1]), int(color[2]))

    #             matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
    #             matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
    #             matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)

    #         plt.figure()
    #         plt.imshow(matches_img)
    #         plt.show()

    #     # Find rigid matrix
    #     if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
    #         H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

    #         # Handle downscale
    #         if self.downscale > 1.0:
    #             H[0, 2] *= self.downscale
    #             H[1, 2] *= self.downscale
    #     else:
    #         print('Warning: not enough matching points')

    #     # Store to next iteration
    #     self.prevFrame = frame.copy()
    #     self.prevKeyPoints = copy.copy(keypoints)
    #     self.prevDescriptors = copy.copy(descriptors)

    #     return H

    def applyFeaures(self, raw_frame, detections=None):
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # find the keypoints
        mask = np.zeros_like(frame)
        # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # Проверка на пустые ключевые точки
        if len(keypoints) == 0:
            # Если ключевых точек нет, сохраняем текущий кадр и возвращаем единичную матрицу
            if self.initializedFirstFrame:
                self.prevFrame = frame.copy()
                self.prevKeyPoints = []
                self.prevDescriptors = None
            return H

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Проверка на пустые дескрипторы после вычисления
        if descriptors is None or len(keypoints) == 0:
            # Если дескрипторы не вычислились, сохраняем текущий кадр
            if self.initializedFirstFrame:
                self.prevFrame = frame.copy()
                self.prevKeyPoints = []
                self.prevDescriptors = None
            return H

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Проверка предыдущих дескрипторов перед сопоставлением
        if self.prevDescriptors is None or len(self.prevDescriptors) == 0:
            # Если предыдущие дескрипторы пустые, обновляем данные и возвращаем единичную матрицу
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Проверка совместимости дескрипторов
        if self.prevDescriptors.dtype != descriptors.dtype:
            print(f'Warning: descriptor type mismatch: {self.prevDescriptors.dtype} vs {descriptors.dtype}')
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Match descriptors.
        try:
            knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
        except cv2.error as e:
            print(f'Warning: matching failed: {e}')
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        # Фильтрация совпадений
        for match_pair in knnMatches:
            # Проверка, что есть как минимум 2 совпадения для сравнения
            if len(match_pair) < 2:
                continue
                
            m, n = match_pair[0], match_pair[1]
            
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        # Проверка на наличие достаточного количества совпадений
        if len(matches) < 4:
            print('Warning: not enough matches for affine transform')
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Draw the keypoint matches on the output image
        if 0:
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)

            plt.figure()
            plt.imshow(matches_img)
            plt.show()

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(currPoints, 0)):
            try:
                H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
                
                # Handle downscale
                if H is not None and self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
                elif H is None:
                    print('Warning: estimateAffinePartial2D failed')
                    H = np.eye(2, 3)
            except Exception as e:
                print(f'Warning: affine estimation failed: {e}')
                H = np.eye(2, 3)
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame, detections=None):

        t0 = time.time()

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        mask = None

        if detections is not None:
            h, w = frame.shape[:2]
            mask = np.zeros_like(frame)
            mask[int(0.02*h):int(0.98*h), int(0.02*w):int(0.98*w)] = 255
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int32)
                x1,y1,x2,y2 = tlbr
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w-1, x2); y2 = min(h-1, y2)
                mask[y1:y2, x1:x2] = 0

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)

        # Проверка на None
        if keypoints is None:
            print('Warning: No keypoints detected')
            keypoints = np.array([], dtype=np.float32).reshape(0, 1, 2)  # Пустой массив вместо None

        # Handle first frame
        if not self.initializedFirstFrame or self.prevKeyPoints is None or len(self.prevKeyPoints) == 0:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        if len(self.prevKeyPoints) == 0 or len(keypoints) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            return H

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        if matchedKeypoints is None or status is None:
            print('Warning: Optical flow failed')
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            return H

        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(currPoints, 0)):
            H_estimated, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC, ransacReprojThreshold=3.0)
            if H_estimated is not None:
                H = H_estimated
                # Handle downscale
                if self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale

                if inliers is not None and len(inliers)>0:
                    self.last_quality = float(np.mean(inliers))
                else:
                    self.last_quality = 0.0

        else:
            print('Warning: not enough matching points')
            self.last_quality = 0.0
            H = np.eye(2,3)


        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        t1 = time.time()

        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)

        # Сглаживание H
        H = self.ema * H + (1.0 - self.ema) * self.prev_H
        self.prev_H = H

        return H

    def applyFile(self, raw_frame, detections=None):
        line = self.gmcFile.readline()
        tokens = line.split("\t")
        H = np.eye(2, 3, dtype=np.float64)
        H[0, 0] = float(tokens[1])
        H[0, 1] = float(tokens[2])
        H[0, 2] = float(tokens[3])
        H[1, 0] = float(tokens[4])
        H[1, 1] = float(tokens[5])
        H[1, 2] = float(tokens[6])

        return H