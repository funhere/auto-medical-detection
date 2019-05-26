#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours

def unmold_mask_2D(mask, bbox, image_shape):
    """Converts a mask into a similar format of it's original shape.
    mask: [height, width]  
    bbox: [y1, x1, y2, x2]
    Returns: a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox
    out_zoom = [y2 - y1, x2 - x1]
    zoom_factor = [i / j for i, j in zip(out_zoom, mask.shape)]
    mask = scipy.ndimage.zoom(mask, zoom_factor, order=1).astype(np.float32)

    full_mask = np.zeros(image_shape[:2])
    full_mask[y1:y2, x1:x2] = mask
    return full_mask



def unmold_mask_3D(mask, bbox, image_shape):
    """Converts a mask into a similar format of it's original shape.
    mask: [height, width]
    bbox: [y1, x1, y2, x2, z1, z2]
    Returns: a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2, z1, z2 = bbox
    out_zoom = [y2 - y1, x2 - x1, z2 - z1]
    zoom_factor = [i/j for i,j in zip(out_zoom, mask.shape)]
    mask = scipy.ndimage.zoom(mask, zoom_factor, order=1).astype(np.float32)

    full_mask = np.zeros(image_shape[:3])
    full_mask[y1:y2, x1:x2, z1:z2] = mask
    return full_mask


def apply_wbc_to_patient(inputs):
    in_patient_results_list, pid, class_dict, wcs_iou, n_ens = inputs
    out_patient_results_list = [[] for _ in range(len(in_patient_results_list))]

    for bix, b in enumerate(in_patient_results_list):

        for cl in list(class_dict.keys()):

            boxes = [(ix, box) for ix, box in enumerate(b) if (box['box_type'] == 'det' and box['box_pred_class_id'] == cl)]
            box_coords = np.array([b[1]['box_coords'] for b in boxes])
            box_scores = np.array([b[1]['box_score'] for b in boxes])
            box_center_factor = np.array([b[1]['box_patch_center_factor'] for b in boxes])
            box_n_overlaps = np.array([b[1]['box_n_overlaps'] for b in boxes])
            box_patch_id = np.array([b[1]['patch_id'] for b in boxes])

            if 0 not in box_scores.shape:
                keep_scores, keep_coords = weighted_box_clustering(
                    np.concatenate((box_coords, box_scores[:, None], box_center_factor[:, None],
                                    box_n_overlaps[:, None]), axis=1), box_patch_id, wcs_iou, n_ens)

                for boxix in range(len(keep_scores)):
                    out_patient_results_list[bix].append({'box_type': 'det', 'box_coords': keep_coords[boxix],
                                             'box_score': keep_scores[boxix], 'box_pred_class_id': cl})

        # add gt boxes back to new output list.
        out_patient_results_list[bix].extend([box for box in b if box['box_type'] == 'gt'])

    return [out_patient_results_list, pid]


def weighted_box_clustering(dets, box_patch_id, thresh, n_ens):
    dim = 2 if dets.shape[1] == 7 else 3
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, -3]
    box_pc_facts = dets[:, -2]
    box_n_ovs = dets[:, -1]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    if dim == 3:
        z1 = dets[:, 4]
        z2 = dets[:, 5]
        areas *= (z2 - z1 + 1)

    # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
    order = scores.argsort()[::-1]

    keep = []
    keep_scores = []
    keep_coords = []

    while order.size > 0:
        i = order[0]  # higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if dim == 3:
            zz1 = np.maximum(z1[i], z1[order])
            zz2 = np.minimum(z2[i], z2[order])
            d = np.maximum(0.0, zz2 - zz1 + 1)
            inter *= d

        # overall between currently highest scoring box and all boxes.
        ovr = inter / (areas[i] + areas[order] - inter)

        # get all the predictions that match the current box to build one cluster.
        matches = np.argwhere(ovr > thresh)

        match_n_ovs = box_n_ovs[order[matches]]
        match_pc_facts = box_pc_facts[order[matches]]
        match_patch_id = box_patch_id[order[matches]]
        match_ov_facts = ovr[matches]
        match_areas = areas[order[matches]]
        match_scores = scores[order[matches]]

        # weight all socres in cluster by patch factors, and size.
        match_score_weights = match_ov_facts * match_areas * match_pc_facts
        match_scores *= match_score_weights

        n_expected_preds = n_ens * np.mean(match_n_ovs)
        n_missing_preds = np.max((0, n_expected_preds - np.unique(match_patch_id).shape[0]))

        denom = np.sum(match_score_weights) + n_missing_preds * np.mean(match_score_weights)

        # compute weighted average score for the cluster
        avg_score = np.sum(match_scores) / denom

        avg_coords = [np.sum(y1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y2[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x2[order[matches]] * match_scores) / np.sum(match_scores)]
        if dim == 3:
            avg_coords.append(np.sum(z1[order[matches]] * match_scores) / np.sum(match_scores))
            avg_coords.append(np.sum(z2[order[matches]] * match_scores) / np.sum(match_scores))

        # filter out the with a conservative threshold, to speed up evaluation.
        if avg_score > 0.01:
            keep_scores.append(avg_score)
            keep_coords.append(avg_coords)

        # get index of all elements that were not matched and discard all others.
        inds = np.where(ovr <= thresh)[0]
        order = order[inds]

    return keep_scores, keep_coords

def merge_preds_per_patient(inputs):
    """
    Applies an adaption of Non-Maximum Surpression per patient.
    Detailed implementation is in nms_bbox.
    @return: 
     results_dict_boxes: 
         list of batch elements. each element is a list of boxes, each box is
         one dictionary: [[box_0, ...], [box_n,...]].
     pid: 
         string. patient id.
    """
    in_patient_results_list, pid, class_dict, merge_iou = inputs
    out_patient_results_list = []

    for cl in list(class_dict.keys()):
        boxes, slice_ids = [], []
        # collect box predictions over batch dimension (slices) and store slice info as slice_ids.
        for bix, b in enumerate(in_patient_results_list):
            det_boxes = [(ix, box) for ix, box in enumerate(b) if
                     (box['box_type'] == 'det' and box['box_pred_class_id'] == cl)]
            boxes += det_boxes
            slice_ids += [bix] * len(det_boxes)

        box_coords = np.array([b[1]['box_coords'] for b in boxes])
        box_scores = np.array([b[1]['box_score'] for b in boxes])
        slice_ids = np.array(slice_ids)

        if 0 not in box_scores.shape:
            keep_ix, keep_z = nms_bbox(
                np.concatenate((box_coords, box_scores[:, None], slice_ids[:, None]), axis=1), merge_iou)
            # Performs soft non-maximum supression
            #keep_ix = utils.non_max_suppression(box_coords, box_scores[:, None], merge_iou)
        else:
            keep_ix, keep_z = [], []

        # store kept predictions in new results list and z-dimension.
        for kix, kz in zip(keep_ix, keep_z):
            out_patient_results_list.append({'box_type': 'det', 'box_coords': list(box_coords[kix]) + kz,
                                             'box_score': box_scores[kix], 'box_pred_class_id': cl})

    out_patient_results_list += [box for b in in_patient_results_list for box in b if box['box_type'] == 'gt']
    out_patient_results_list = [out_patient_results_list] # add dummy batch dimension 1 for 3D.

    return [out_patient_results_list, pid]

def nms_bbox(dets, thresh):
    """
    An adaptation of Non-maximum surpression.

    :param dets: (n_detections, (y1, x1, y2, x2, scores, slice_id)
    :param thresh: iou threshold.
    :return: keep: (n_keep) 1D tensor of indices to be kept.
    :return: keep_z: (n_keep, [z1, z2]) z-axis to be added to boxes.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, -2]
    slice_id = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    keep_z = []

    while order.size > 0:  # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
        i = order[0]  # pop higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order] - inter)
        matches = np.argwhere(ovr > thresh)  # get all the elements that match the current box and have a lower score

        slice_ids = slice_id[order[matches]]
        core_slice = slice_id[int(i)]
        upper_wholes = [ii for ii in np.arange(core_slice, np.max(slice_ids)) if ii not in slice_ids]
        lower_wholes = [ii for ii in np.arange(np.min(slice_ids), core_slice) if ii not in slice_ids]
        max_valid_slice_id = np.min(upper_wholes) if len(upper_wholes) > 0 else np.max(slice_ids)
        min_valid_slice_id = np.max(lower_wholes) if len(lower_wholes) > 0 else np.min(slice_ids)
        z_matches = matches[(slice_ids <= max_valid_slice_id) & (slice_ids >= min_valid_slice_id)]

        z1 = np.min(slice_id[order[z_matches]]) - 1
        z2 = np.max(slice_id[order[z_matches]]) + 1

        keep.append(i)
        keep_z.append([z1, z2])
        order = np.delete(order, z_matches, axis=0)

    return keep, keep_z



def create_csv_output(cf, logger, results_list):
    """
    Output format is one line per patient:
       PatientID score pred_class x y w h score pred_class x y w h .....
    :param results_list: 
        [[patient_results, patient_id], [patient_results, patient_id], ...]
    """
    logger.info('creating csv output file at {}'.format(os.path.join(cf.exp_dir, 'output.csv')))
    pred_df = pd.DataFrame(columns=['patientID', 'PredictionString'])
    for r in results_list:
        pid = r[1]
        prediction_string = ''
        for box in r[0][0]:
            coords = box['box_coords']
            score = box['box_score']
            pred_class = box['box_pred_class_id']

            if score >= cf.min_det_thresh:
                x = coords[1] #* cf.pp_downsample_factor
                y = coords[0] #* cf.pp_downsample_factor
                width = (coords[3] - coords[1]) #* cf.pp_downsample_factor
                height = (coords[2] - coords[0]) #* cf.pp_downsample_factor
                if len(coords) == 6:
                    z = coords[4]
                    depth = (coords[5] - coords[4])
                    prediction_string += '{} {} {} {} {} {} {} {}'.format(score, pred_class, x, y, z, width, height, depth)
                else:
                    prediction_string += '{} {} {} {} {} {} '.format(score, pred_class, x, y, width, height)

        if prediction_string == '':
            prediction_string = None
        pred_df.loc[len(pred_df)] = [pid, prediction_string]
    pred_df.to_csv(os.path.join(cf.exp_dir, 'output.csv'), index=False)


def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    if np.sum(union)>0:
        overlaps = intersections / union
    else:
        return 0
    return overlaps

def compute_mean_iou(overlaps, list_all=False):
    miou = []
    overlaps *= (overlaps>=.5)
    for step in np.arange(0.5,1,0.05):
        overlap_step = overlaps >=step
        tp = np.sum(np.sum(overlap_step,axis=0)>0)
        fp = np.sum(np.sum(overlap_step,axis=0)==0)
        fn = np.sum(np.sum(overlap_step,axis=-1)==0)
        miou.append(tp/(tp+fp+fn))
    if list_all:
        return miou
    else:
        return np.mean(miou)

def compute_mask_scores(overlaps, list_all=False):
    miou, recalls, precisions = [], [], []
    overlaps *= (overlaps>=.5)
    for step in np.arange(0.5,1,0.05):
        overlap_step = overlaps >=step
        tp = np.sum(np.sum(overlap_step,axis=0)>0)
        fp = np.sum(np.sum(overlap_step,axis=0)==0)
        fn = np.sum(np.sum(overlap_step,axis=-1)==0)
        miou.append(tp/(tp+fp+fn))
        recalls.append(tp/(tp+fn))
        precisions.append(tp/(tp+fp))
    if list_all:
        return miou,recalls, precisions 
    else:
        return np.mean(miou), np.mean(recalls), np.mean(precisions)

def mask_12m(mask, cval = 1):
    index = mask!=0
    nb_mask = mask[index].max() 
    if nb_mask ==0 :
        return np.zeros(mask.shape + (1,), dtype='uint8')
    else:
        mask_full = np.zeros(mask.shape+(nb_mask,), dtype='uint8') 
        mask_full[index, mask[index]-1]=cval
        return mask_full

def mask_12m_no(mask, vals=None, cval = 1):
    if vals is None:
        vals = np.unique(mask) 
        vals = vals[vals>0]
    index = mask!=0
    nb_mask = len(vals) 
    inds = np.arange(vals.max()+1)
    inds[vals] = np.arange(nb_mask)
    if nb_mask ==0 :
        return np.zeros(mask.shape + (1,), dtype='uint8')
    else:
        mask_full = np.zeros(mask.shape+(nb_mask,), dtype='uint8') 
        mask_full[index, inds[mask[index]]]=cval
        return mask_full
    
def get_score(base_label, mask, list_all = False):
    if (mask.max()==0)|(base_label.max()==0):
        if (mask.max()==0)&(base_label.max()==0):
            return 1
        else:
            return 0
    mask_pred = mask_12m_no(base_label)
    mask_true = mask_12m(mask)    
    overlaps = compute_overlaps_masks(mask_true, mask_pred)
    return compute_mask_scores(overlaps, list_all = list_all)

def plot_boundary(image, true_masks=None, pred_masks=None, ax=None):
    """
    Plots provided boundaries for a given image.
    """
    if ax is None:
        n_rows = 1
        n_cols = 1

        fig = plt.figure(figsize=[4*n_cols, int(4*n_rows)])
        gs = gridspec.GridSpec(n_rows, n_cols)

        ax = fig.add_subplot(gs[0])

    ax.imshow(image[:,:,0])
    if true_masks is not None:
        for i in range(true_masks.shape[-1]):
            contours = find_contours(true_masks[..., i], 0.5, fully_connected='high')

            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
    if pred_masks is not None:
        for i in range(pred_masks.shape[-1]):
            contours = find_contours(pred_masks[..., i], 0.5, fully_connected='high')

            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)  # aspect ratio of 1
