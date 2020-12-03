import sys
CENTERNET_PATH = './lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
import argparse
import glob
import os
import cv2
import json
import numpy as np
from scipy.spatial import distance
from baseline.pixelize import pix
from baseline.blur import blurry


def visual_pose(img, points):
    edges = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
            [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
    ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 255),
            (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 255),
            (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
    colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255)]

    points = np.array(points, dtype=np.int32).reshape(17, 2)
    for j in range(17):
        cv2.circle(img, (int(points[j, 0]), int(points[j, 1])), 3, colors_hp[j], -1)
    for j, e in enumerate(edges):
        if points[e].min() > 0:
            cv2.line(img, (int(points[e[0], 0]), int(points[e[0], 1])), (int(points[e[1], 0]), int(points[e[1], 1])), ec[j], 2, lineType=cv2.LINE_AA)

def calculate_iou(detection, reidenf):
    x_left = max(reidenf[0], detection[0])
    y_top = max(reidenf[1], detection[1])
    x_right = min(reidenf[2], detection[2])
    y_bottom = min(reidenf[3], detection[3])

    if x_right < x_left or y_bottom < y_top:
        return 0

    intersection_area = abs(x_right - x_left) * abs(y_bottom - y_top)
    bb1_area = abs(reidenf[0] - reidenf[2]) * abs(reidenf[1] - reidenf[3])
    bb2_area = abs(detection[0] - detection[2]) * abs(detection[1] - detection[3])
    ioue = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return ioue

def test_pose_single(setting):
    
    # Testing parameter
    MODEL_PATH = setting['model_path']
    TASK = 'multi_pose'
    NET = setting['network']
    opt = opts().init('{} --load_model {} --arch {} --nms --down_ratio 4 --gpus 2'.format(TASK, MODEL_PATH, NET).split(' '))

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    # Baseline testing
    if setting['pix']:
        print("____________Baseline Pixelization Training______________")
    if setting['blur']:
        print("____________Baseline Blur Training______________")

    # Detector
    detector = detector_factory[opt.task](opt)
    color = [(255, 0, 0), (0, 255, 0), (0, 255, 255)]

    # GT
    labels = json.load(open(setting['label_path']))
    gts = {}
    for anno in labels['annotations']:
        img_id = anno['image_id']
        try:
            if img_id in list(gts.keys()):
               gts[img_id].append({'bbox': anno['bbox'],
                                'keypoints': anno['keypoints']})
            else:
               gts[img_id] = [{'bbox': anno['bbox'],
                                'keypoints': anno['keypoints']}]
        except:
           continue

    # Testing
    img_list = labels['images']
    dist_keypoints = []
    iou_bboxs = []
    _idx = 0
    visualize_index = 0
    total_gt = 0
    
    # stat for pck
    pck_stats = [[] for _ in range(5)]
    pck_stat_filenames = [[] for _ in range(5)]

    for img in img_list:
        img_id = img['id']
        img_path = os.path.join(setting['data_path'], img['file_name'])
        print(img_path)
        ret = detector.run(img_path)['results']
        ori_img = cv2.imread(img_path)

        #ori_img = 255 - ori_img
        # Baseline
        if opt.basepix:
            ori_img = pix(ori_img)
        if opt.baseblur:
            ori_img = blurry(ori_img)
        
        gt_img = ori_img.copy()
        pred_img = ori_img.copy()

        cv2.putText(gt_img, "GT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(pred_img, "Pred", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        pred_keypoints = []
        pred_bboxs = []
        for r in ret[1]:
            if r[4] > opt.vis_thresh:
                pt1 = (int(r[0]), int(r[1]))
                pt2 = (int(r[2]), int(r[3]))
                pred_img = cv2.rectangle(pred_img, pt1, pt2, (0, 0, 200), 2)
                pred_bboxs.append(r[:4])
                print(r[:4])
                pred_keypoint = []
                for i in range(5, 39, 2):
                    pred_keypoint.append((r[i], r[i+1]))
                pred_keypoints.append(pred_keypoint)
        try:
            gt = gts[img_id]
        except:
            print('empty annotation, img_path: ', img_path)
            gt = []
            #continue
        total_gt += len(gt)
        for k in gt:
            gt_bbox = k['bbox']
            # convert (x1, y1, w, h) => (x1, y2, x2, y2)
            gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
            pt1 = (int(gt_bbox[0]), int(gt_bbox[1]))
            pt2 = (int(gt_bbox[2]), int(gt_bbox[3]))
            gt_img = cv2.rectangle(gt_img, pt1, pt2, (0, 255, 0), 2)

            gt_keypoint = []
            gt_keypoint_mask = []

            for idx in range(len(k['keypoints'])):
                if idx % 3 == 0:
                    if k['keypoints'][idx+2] == 0:
                        gt_keypoint_mask.append(-1)
                    else:
                        gt_keypoint_mask.append(1)
                    gt_keypoint.append((k['keypoints'][idx], k['keypoints'][idx+1]))
            if len(gt_keypoint) != 17:
                break
            if setting['visualize']:
                visual_pose(gt_img, gt_keypoint)
            np_gt_keypoint = np.array(gt_keypoint)
            np_pred_bboxs = np.array(pred_bboxs)
            np_gt_bbox = np.array(gt_bbox)
            np_gt_keypoint_mask = np.array(gt_keypoint_mask)
            # torso_dist = |left_shoulder - right_hip|, 1e-8 avoid zero division
            torso_dist = distance.euclidean(np_gt_keypoint[5], np_gt_keypoint[12]) + 1e-8

            # False Negative
            if len(pred_bboxs) == 0:
                d = np.ones(17) * np.inf
                dist_keypoints.append(d)
                iou = -2
                iou_bboxs.append(iou)
            else:
                # greedly match human bounding box
                ious = [calculate_iou(pred_bbox, gt_bbox) for pred_bbox in pred_bboxs]
                target_id = int(np.argmax(ious))

                pred_keypoint = pred_keypoints[target_id]
                pred_bbox = pred_bboxs[target_id]
                pred_iou = ious[target_id]

                if setting['visualize']:
                    visual_pose(pred_img, pred_keypoint)
                # calculate the keypoint distance and normalized by torso_dist
                d = np.sqrt(np.sum(np.square(np.subtract(gt_keypoint, pred_keypoint)), axis=-1)) / torso_dist
                d = d * np_gt_keypoint_mask
                dist_keypoints.append(d)
                iou_bboxs.append(pred_iou)

                del pred_keypoints[target_id]
                del pred_bboxs[target_id]

        if not gt: 
            # False Positive
            for _ in range(len(pred_bboxs)):
                d = np.ones(17) * np.inf
                dist_keypoints.append(d)
                iou = -1
                iou_bboxs.append(iou)
        
        # visualize
        if setting['visualize']:
            cv2.imshow('img', pred_img)
            show_pred = True
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if show_pred:
                        cv2.imshow('img', gt_img)
                    else:
                        cv2.imshow('img', pred_img)
                    show_pred = not show_pred
                elif key == ord('s'):
                    cv2.imwrite("./vis/" + str(visualize_index) + "_pred.png", pred_img)
                    cv2.imwrite("./vis/" + str(visualize_index) + "_gt.png", gt_img)
                    visualize_index += 1
                elif key == ord('n'):
                    break
        
        # calculate pck for current image
        pck_indi_correct = 0
        pck_indi_count = 0
        pck_indi_all = 0
        if gt:
            pck_indi_count = len(gt)
            pck_indi_all = len(gt) * 17
        else:
            pck_indi_count = len(pred_bboxs)
            pck_indi_all = len(pred_bboxs) * 17
        if pck_indi_all == 0: continue
        for dist_keypoint in dist_keypoints[-pck_indi_count:]:
            for idx, dist in enumerate(dist_keypoint):
                if dist < 0:
                    pck_indi_all -= 1
                elif dist <= setting['pck_threshold']:
                    pck_indi_correct += 1
        pck_indi = pck_indi_correct / pck_indi_all
        print('current PCK = {}'.format(pck_indi))
        if 0 <= pck_indi < 0.2:
            pck_stats[0].append(pck_indi)
            pck_stat_filenames[0].append(img['file_name'])
            # cv2.imwrite("../../data/images/0510_user04_tmp/2/" + img['file_name'].split('/')[-1], pred_img)
        elif 0.2 <= pck_indi < 0.4:
            pck_stats[1].append(pck_indi)
            pck_stat_filenames[1].append(img['file_name'])
            # cv2.imwrite("../../data/images/0510_user04_tmp/4/" + img['file_name'].split('/')[-1], pred_img)
        elif 0.4 <= pck_indi < 0.6:
            pck_stats[2].append(pck_indi)
            pck_stat_filenames[2].append(img['file_name'])
            # cv2.imwrite("../../data/images/0510_user04_tmp/6/" + img['file_name'].split('/')[-1], pred_img)
        elif 0.6 <= pck_indi < 0.8:
            pck_stats[3].append(pck_indi)
            pck_stat_filenames[3].append(img['file_name'])
            # cv2.imwrite("../../data/images/0510_user04_tmp/8/" + img['file_name'].split('/')[-1], pred_img)
        elif 0.8 <= pck_indi <= 1.0:
            pck_stats[4].append(pck_indi)
            pck_stat_filenames[4].append(img['file_name'])
            # cv2.imwrite("../../data/images/0510_user04_tmp/10/" + img['file_name'].split('/')[-1], pred_img)
        else:
            raise "PCK larger than 1.0"
    
    pck_correct = [0 for _ in range(17)]
    bbox_tp = 0
    bbox_fp = 0
    bbox_fn = 0
    
    num_keypoint_samples = [len(dist_keypoints) for _ in range(17)]
    
    for dist_keypoint, iou_bbox in zip(dist_keypoints, iou_bboxs):

        if iou_bbox >= setting['iou_threshold']:
            bbox_tp += 1
        elif iou_bbox == -1:
            bbox_fp += 1
        elif iou_bbox == -2:
            bbox_fn += 1
        elif iou_bbox >= 0:
            bbox_fn += 1
            bbox_fp += 1
        else:
            raise('IOU value error')

        for idx, dist in enumerate(dist_keypoint):
            if dist < 0:
                num_keypoint_samples[idx] -= 1
            elif dist <= setting['pck_threshold']:
                pck_correct[idx] += 1

    print('tp, fp, fn, gt = ', bbox_tp, bbox_fp, bbox_fn, total_gt)
    
    for idx, (stats, filenames) in enumerate(zip(pck_stats, pck_stat_filenames)):
        print("len(stats): {}".format(len(stats)))
        print("len(filenames): {}".format(len(filenames)))
        for stat, filename in zip(stats, filenames):
            print('filename: {}, PCK: {}'.format(filename, stat))

    pck_correct = [p/n for p, n in zip(pck_correct, num_keypoint_samples)]
    bbox_precision = bbox_tp / (bbox_tp + bbox_fp) if bbox_tp + bbox_fp != 0 else -1
    bbox_recall = bbox_tp / (bbox_tp + bbox_fn) if bbox_tp + bbox_fn != 0 else -1

    output = {'bbox_tp': bbox_tp, 'bbox_fp': bbox_fp, 'bbox_fn': bbox_fn, 'total_gt': total_gt, 'precision': bbox_precision, 'recall': bbox_recall, 'pck_correct': pck_correct, 'PCK_all': sum(pck_correct)/len(pck_correct), 'PCK_action': (pck_correct[5]+pck_correct[6]+pck_correct[11]+pck_correct[12]+pck_correct[13]+pck_correct[14])/6}
    return output


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', help='load model pth')
    parser.add_argument('--pck_threshold', dest='pck_threshold', type=float, default=0.2, help='pck threshold')
    parser.add_argument('--iou_threshold', dest='iou_threshold', type=float,  default=0.5, help='iou threshold')
    parser.add_argument('--data_path', dest='data_path', default='../../data/images/', help='load data dir')
    parser.add_argument('--label_path', dest='label_path', help='load data dir')
    parser.add_argument('--visualize', dest='visualize', default=False, help='load data dir')
    parser.add_argument('--pix', default=0, help='testing for baseline pixelization')
    parser.add_argument('--blur', default=0, help='testing for baseline blur')
    parser.add_argument('--network', default='hardnet', help='testing network')
    args = parser.parse_args()

    setting = {'model_path': args.model_path, 'pck_threshold': args.pck_threshold, 'iou_threshold': args.iou_threshold, 'data_path': args.data_path, 'label_path': args.label_path, 'visualize': args.visualize, 'pix': args.pix, 'blur': args.blur, 'network': args.network}

    output = test_pose_single(setting)
    print('tp: ', output['bbox_tp'], ' fp: ', output['bbox_fp'], ' fn: ', output['bbox_fn'], ' total gt: ', output['total_gt'])

    print("Keypoints: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle")
    print('PCK keypoints: {}' .format(output['pck_correct']))
    print('PCK all, BBOX precision, recall: {} {} {}' .format(output['PCK_all'], output['precision'], output['recall']))
    print('PCK action points: {}'.format(output['PCK_action']))

