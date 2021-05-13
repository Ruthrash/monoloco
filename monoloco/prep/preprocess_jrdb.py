# pylint: disable=too-many-statements, too-many-branches, too-many-nested-blocks

"""Preprocess annotations with KITTI ground-truth"""

from monoloco.visuals.pifpaf_show import boxes
import os
import glob
import copy
import math
import logging
from collections import defaultdict
import json
import warnings
import datetime
from PIL import Image

import torch

from .. import __version__
from ..utils import split_training, split_training_jrdb,  get_iou_matches, get_iou_matches_jrdb, append_cluster, get_calibration, open_features, open_annotations, \
    extract_stereo_matches, make_new_directory, \
    check_conditions, to_spherical, correct_angle
from ..network.process import preprocess_pifpaf, preprocess_monoloco
from .transforms import flip_inputs, flip_labels, height_augmentation


class PreprocessJRDB:
    """Prepare arrays with same format as nuScenes preprocessing but using ground truth txt files"""

    # KITTI Dataset files
    #dir_gt = os.path.join('data', 'jrdb', 'gt')
    dir_gt = '/home/ruthz/cvpr_challenge/JRDB/kitti_labels'

    # SOCIAL DISTANCING PARAMETERS
    THRESHOLD_DIST = 2  # Threshold to check distance of people
    RADII = (0.3, 0.5, 1)  # expected radii of the o-space
    SOCIAL_DISTANCE = True

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dic_jo = {
        'train': dict(features=[], predictions=[], labels=[], names=[] ),
        'val': dict(features=[], predictions=[], labels=[], names=[] ),
        'test': dict(features=[], predictions=[], labels=[], names=[] ),
        'version': __version__,
    }

    dic_names = defaultdict(lambda: defaultdict(list))
    dic_std = defaultdict(lambda: defaultdict(list))
    categories_gt = dict(train=['Pedestrian', 'Person_sitting'], val=['Pedestrian'])

    def __init__(self, dir_ann, sample=False):
        self.dir_ann = dir_ann

        self.sample = sample

        assert os.path.isdir(self.dir_ann), "Annotation directory not found"
        assert any(os.scandir(self.dir_ann)), "Annotation directory empty"
        assert os.path.isdir(self.dir_gt), "Ground truth directory not found"
        assert any(os.scandir(self.dir_gt)), "Ground-truth directory empty"

        #print(self.names_gt) 
        self.seq_list_gt = glob.glob(self.dir_gt+"/*")
        self.names_gt = glob.glob(self.dir_gt+"/*/*.txt", recursive=True)
        self.names_gt.sort()
        self.names_gt = tuple(self.names_gt)

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        dir_out = os.path.join('data', 'arrays')
        dir_out = "/media/ruthz/DATA/data/arrays"
        # * output 
        self.path_joints = os.path.join(dir_out, 'joints-kitti-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-kitti-' + now_time + '.json')

        path_train = os.path.join('splits', 'jrdb_train.txt')
        path_val = os.path.join('splits', 'jrdb_val.txt')
        #self.set_train, self.set_val = split_training(self.names_gt, path_train, path_val)
        
        base_dir = self.dir_gt   
        self.set_train, self.set_val = split_training_jrdb(base_dir, path_train, path_val)
        #print(self.set_train, self.set_val)
        self.phase, self.name = None, None
        self.stats = defaultdict(int)
        self.stats_stereo = defaultdict(int)
        self.iou_min = 0.3
        

    def run(self):
        # self.names_gt = ('002282.txt',)
        for self.name in self.names_gt:##each ground truth file 
            #print(self.name)
            # Extract ground truth
            path_gt = self.name
            basename = self.name.split('.')[0]
            basenames = basename.split('/')
            basename = basenames[-2]+"/"+basenames[-1]
            self.phase, file_not_found = self._factory_phase(self.name)##split training and validation
            category = 'all' if self.phase == 'train' else 'pedestrian'
            if file_not_found:
                self.stats['fnf'] += 1
                continue

            # * path_gt should be individual ground truth 
            boxes_gt, labels, truncs_gt, occs_gt, _ = parse_ground_truth(path_gt, category=category, spherical=False)

            #self.stats['gt_' + self.phase] += len(boxes_gt)
            #self.stats['gt_files'] += 1
            #self.stats['gt_files_ped'] += min(len(boxes_gt), 1)  # if no boxes 0 else 1
            
            self.dic_names[basename + '.jpg']['boxes'] = copy.deepcopy(boxes_gt)
            self.dic_names[basename + '.jpg']['labels'] = copy.deepcopy(labels)
            self.dic_names[basename + '.jpg']['truncs'] = copy.deepcopy(truncs_gt)
            self.dic_names[basename + '.jpg']['occs'] = copy.deepcopy(occs_gt)

            # Extract annotations along with features
            dic_annotations = self.parse_annotations(boxes_gt=boxes_gt, labels=labels, truncs_gt=truncs_gt, occs_gt=occs_gt, basename=basename)

            #assert len(list(dic_annotations.values())) == len(boxes_gt), print("missing features for ground truth")
            #print('truth', dic_annotations)


            # Match each feature with a ground truth
            matches = get_iou_matches_jrdb(dic_annotations, self.iou_min)
            self._process_annotation_fpointnet(matches, dic_annotations)

        with open(self.path_joints, 'w') as file:
            json.dump(self.dic_jo, file)
        with open(os.path.join(self.path_names), 'w') as file:
            json.dump(self.dic_names, file)
        # self._cout()

    def parse_annotations(self, boxes_gt, labels, truncs_gt, occs_gt, basename):

        basenames = basename.split('/')
        path_ann = os.path.join(self.dir_ann, basenames[0]+'/features_3d/'+basenames[1]+ '.yaml')
        annotations = dict()
        annotations['features'] = open_features(path_ann)
        annotations['gt'] = boxes_gt 
        annotations['labels'] = labels #+ truncs_gt + occs_gt
        annotations['truncs'] = truncs_gt
        annotations['occs'] = occs_gt
        return annotations#, kk, tt

    def _process_annotation_fpointnet(self, matches, dic_annotations):
        for match in matches: 
            print(self.name,match[0], match[1])
            self.dic_jo[self.phase]['features'].append(dic_annotations['features'][match[0]]['feature'])#from fpointnet
            prediction = [dic_annotations['features'][match[0]]['x'],
                        dic_annotations['features'][match[0]]['y'],
                        dic_annotations['features'][match[0]]['z'],
                        dic_annotations['features'][match[0]]['l'],
                        dic_annotations['features'][match[0]]['w'],
                        dic_annotations['features'][match[0]]['h']
                        ]
            self.dic_jo[self.phase]['predictions'].append(prediction)#from fpointnet
            self.dic_jo[self.phase]['labels'].append(dic_annotations['labels'][match[1]]  )#from groundtruth 
            self.dic_jo[self.phase]['names'].append(self.name)
        return

    def _process_annotation_mono(self, kp, kk, label):
        """For a single annotation, process all the labels and save them"""
        kp = kp.tolist()
        inp = preprocess_monoloco(kp, kk).view(-1).tolist()

        # Save
        self.dic_jo[self.phase]['kps'].append(kp)
        self.dic_jo[self.phase]['X'].append(inp)
        self.dic_jo[self.phase]['Y'].append(label)
        self.dic_jo[self.phase]['names'].append(self.name)  # One image name for each annotation
        append_cluster(self.dic_jo, self.phase, inp, label, kp)
        self.stats['total_' + self.phase] += 1

    def _process_annotation_stereo(self, kp, kk, label, kps_r):
        """For a reference annotation, combine it with some (right) annotations and save it"""

        zz = label[2]
        stereo_matches, cnt_amb = extract_stereo_matches(kp, kps_r, zz,
                                                         phase=self.phase,
                                                         seed=self.stats_stereo['pair'])
        self.stats_stereo['ambiguous'] += cnt_amb

        for idx_r, s_match in stereo_matches:
            label_s = label + [s_match]  # add flag to distinguish "true pairs and false pairs"
            self.stats_stereo['true_pair'] += 1 if s_match > 0.9 else 0
            self.stats_stereo['pair'] += 1  # before augmentation

            # ---> Remove noise of very far instances for validation
            # if (self.phase == 'val') and (label[3] >= 50):
            #     continue

            #  ---> Save only positives unless there is no positive (keep positive flip and augm)
            # if num > 0 and s_match < 0.9:
            #     continue

            # Height augmentation
            flag_aug = False
            if self.phase == 'train' and 3 < label[2] < 30 and (s_match > 0.9 or self.stats_stereo['pair'] % 2 == 0):
                flag_aug = True

            # Remove height augmentation
            # flag_aug = False

            if flag_aug:
                kps_aug, labels_aug = height_augmentation(kp, kps_r[idx_r:idx_r + 1], label_s,
                                                          seed=self.stats_stereo['pair'])
            else:
                kps_aug = [(kp, kps_r[idx_r:idx_r + 1])]
                labels_aug = [label_s]

            for i, lab in enumerate(labels_aug):
                assert len(lab) == 11, 'dimensions of stereo label is wrong'
                self.stats_stereo['pair_aug'] += 1
                (kp_aug, kp_aug_r) = kps_aug[i]
                input_l = preprocess_monoloco(kp_aug, kk).view(-1)
                input_r = preprocess_monoloco(kp_aug_r, kk).view(-1)
                keypoint = torch.cat((kp_aug, kp_aug_r), dim=2).tolist()
                inp = torch.cat((input_l, input_l - input_r)).tolist()
                self.dic_jo[self.phase]['kps'].append(keypoint)
                self.dic_jo[self.phase]['X'].append(inp)
                self.dic_jo[self.phase]['Y'].append(lab)
                self.dic_jo[self.phase]['names'].append(self.name)  # One image name for each annotation
                append_cluster(self.dic_jo, self.phase, inp, lab, keypoint)
                self.stats_stereo['total_' + self.phase] += 1  # including height augmentation

    def _cout(self):
        print('-' * 100)
        print(f"Number of GT files: {self.stats['gt_files']} ")
        print(f"Files with at least one pedestrian/cyclist: {self.stats['gt_files_ped']}")
        print(f"Files not found: {self.stats['fnf']}")
        print('-' * 100)
        our = self.stats['match'] - self.stats['flipping_match']
        gt = self.stats['gt_train'] + self.stats['gt_val']
        print(f"Ground truth matches: {100 * our  / gt:.1f} for left images (train and val)")
        print(f"Parsed instances: {self.stats['instances']}")
        print(f"Ground truth instances: {gt}")
        print(f"Matched instances: {our}")
        print(f"Including horizontal flipping: {self.stats['match']}")

        if self.mode == 'stereo':
            print('-' * 100)
            print(f"Ambiguous instances removed: {self.stats_stereo['ambiguous']}")
            print(f"True pairs ratio: {100 * self.stats_stereo['true_pair'] / self.stats_stereo['pair']:.1f}% ")
            print(f"Height augmentation pairs: {self.stats_stereo['pair_aug'] - self.stats_stereo['pair']} ")
            print('-' * 100)
        total_train = self.stats_stereo['total_train'] if self.mode == 'stereo' else self.stats['total_train']
        total_val = self.stats_stereo['total_val'] if self.mode == 'stereo' else self.stats['total_val']
        print(f"Total annotations for TRAINING: {total_train}")
        print(f"Total annotations for VALIDATION: {total_val}")
        print('-' * 100)
        print(f"\nOutput files:\n{self.path_names}\n{self.path_joints}")
        print('-' * 100)

    def process_activity(self):
        """Augment ground-truth with flag activity"""

        from monoloco.activity import social_interactions  # pylint: disable=import-outside-toplevel
        main_dir = os.path.join('data', 'kitti')
        dir_gt = os.path.join(main_dir, 'gt')
        dir_out = os.path.join(main_dir, 'gt_activity')
        make_new_directory(dir_out)
        cnt_tp, cnt_tn = 0, 0

        # Extract validation images for evaluation
        category = 'pedestrian'

        for name in self.set_val:
            # Read
            path_gt = os.path.join(dir_gt, name)
            _, ys, _, _, lines = parse_ground_truth(path_gt, category, spherical=False)
            angles = [y[10] for y in ys]
            dds = [y[4] for y in ys]
            xz_centers = [[y[0], y[2]] for y in ys]

            # Write
            path_out = os.path.join(dir_out, name)
            with open(path_out, "w+") as ff:
                for idx, line in enumerate(lines):
                    if social_interactions(idx, xz_centers, angles, dds,
                                           n_samples=1,
                                           threshold_dist=self.THRESHOLD_DIST,
                                           radii=self.RADII,
                                           social_distance=self.SOCIAL_DISTANCE):
                        activity = '1'
                        cnt_tp += 1
                    else:
                        activity = '0'
                        cnt_tn += 1

                    line_new = line[:-1] + ' ' + activity + line[-1]
                    ff.write(line_new)

        print(f'Written {len(self.set_val)} new files in {dir_out}')
        print(f'Saved {cnt_tp} positive and {cnt_tn} negative annotations')

    def _factory_phase(self, name):
        """Choose the phase"""
        phase = None
        flag = False
        if name in self.set_train:
            phase = 'train'
        elif name in self.set_val:
            phase = 'val'
        else:
            flag = True
        return phase, flag


def parse_ground_truth(path_gt, category, spherical=False):
    """Parse KITTI ground truth files"""

    boxes_gt = []
    labels = []
    truncs_gt = []  # Float from 0 to 1
    occs_gt = []  # Either 0,1,2,3 fully visible, partly occluded, largely occluded, unknown
    lines = []
    print("gt", path_gt)
    with open(path_gt, "r") as f_gt:
        for line_gt in f_gt:
            line = line_gt.split()
            if not check_conditions(line_gt, category, method='gt'):
                continue
            truncs_gt.append(float(line[1]))##
            occs_gt.append(int(line[2]))##
            boxes_gt.append([float(x) for x in line[5:9]])##
            xyz = [float(x) for x in line[12:15]]###
            hwl = [float(x) for x in line[9:12]]
            dd = float(math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))
            yaw = float(line[15])
            ##jrdbcorrections
            #assert - math.pi <= yaw <= math.pi
            alpha = float(line[4])
            sin, cos, yaw_corr = correct_angle(yaw, xyz)
            ##jrdbcorrections
            #print(min(abs(-yaw_corr - alpha), (abs(yaw_corr - alpha))))
            #assert min(abs(-yaw_corr - alpha), (abs(yaw_corr - alpha))) < 0.15, "more than 10 degrees of error"
            if spherical:
                rtp = to_spherical(xyz)
                loc = rtp[1:3] + xyz[2:3] + rtp[0:1]  # [theta, psi, z, r]
            else:
                loc = xyz + [yaw] + [dd]
            cat = line[0]  # 'Pedestrian', or 'Person_sitting' for people
            output = loc + hwl + [sin, cos, yaw, cat] +[line[1]]+[line[2]]

            labels.append(output)
            lines.append(line_gt)
    return boxes_gt, labels, truncs_gt, occs_gt, lines


def factory_file(dir_ann, basename):#, ann_type='left'):
    """Choose the annotation files"""
    basenames = basename.split('/')
    path_ann = os.path.join(dir_ann, basenames[0]+'/features_3d/'+basenames[1]+ '.yaml')
    annotations = open_features(path_ann)
    return annotations#, kk, tt