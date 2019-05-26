#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def weighted_box_clustering(dets, box_patch_id, thresh, n_ens):
    """
    consolidates overlapping predictions resulting from patch overlaps, test data augmentations and temporal ensembling.
    clusters predictions together with iou > thresh (like in NMS). 
    :param dets: (n_dets, (y1, x1, y2, x2, (z1), (z2), scores, box_pc_facts, box_n_ovs)
    :param thresh: threshold for iou_matching.
    :param n_ens: number of models, that are ensembled. (-> number of expected predicitions per position)
    :return: keep_scores: (n_keep)  new scores of boxes to be kept.
    :return: keep_coords: (n_keep, (y1, x1, y2, x2, (z1), (z2)) new coordinates of boxes to be kept.
    """
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

        # the number of missing predictions is obtained as the number of patches,
        # which did not contribute any prediction to the current cluster.
        n_missing_preds = np.max((0, n_expected_preds - np.unique(match_patch_id).shape[0]))

        # missing preds are given the mean weighting
        denom = np.sum(match_score_weights) + n_missing_preds * np.mean(match_score_weights)

        # compute weighted average score for the cluster
        avg_score = np.sum(match_scores) / denom

        # compute weighted average of coordinates for the cluster. now only take existing
        # predictions into account.
        avg_coords = [np.sum(y1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y2[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x2[order[matches]] * match_scores) / np.sum(match_scores)]
        if dim == 3:
            avg_coords.append(np.sum(z1[order[matches]] * match_scores) / np.sum(match_scores))
            avg_coords.append(np.sum(z2[order[matches]] * match_scores) / np.sum(match_scores))

        # some clusters might have very low scores due to high amounts of missing predictions.
        # filter out the with a conservative threshold, to speed up evaluation.
        if avg_score > 0.01:
            keep_scores.append(avg_score)
            keep_coords.append(avg_coords)

        # get index of all elements that were not matched and discard all others.
        inds = np.where(ovr <= thresh)[0]
        order = order[inds]

    return keep_scores, keep_coords



def nms_2to3D(dets, thresh):
    """
    Merges 2D boxes to 3D cubes. 

    :param dets: (n_detections, (y1, x1, y2, x2, scores, slice_id)
    :param thresh: iou matchin threshold (like in NMS).
    :return: keep: (n_keep) 1D tensor of indices to be kept.
    :return: keep_z: (n_keep, [z1, z2]) z-coordinates to be added to boxes
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



def get_mirrored_patch_crops(patch_crops, org_img_shape):
    """
    apply 3 mirrror transformations (x-axis, y-axis, x&y-axis)
   
    :param patch_crops: list of crops: each element is a list of coordinates for given crop [[y1, x1, ...], [y1, x1, ..]]
    :param org_img_shape: shape of patient volume used as world coordinates.
    :return: list of mirrored patch crops: lenght=3. each element is a list of transformed patch crops.
    """
    mirrored_patch_crops = []

    # y-axis transform.
    mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  ii[2], ii[3]] if len(ii) == 4 else
                                 [org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  ii[2], ii[3], ii[4], ii[5]] for ii in patch_crops])

    # x-axis transform.
    mirrored_patch_crops.append([[ii[0], ii[1],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2]] if len(ii) == 4 else
                                 [ii[0], ii[1],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2],
                                  ii[4], ii[5]] for ii in patch_crops])

    # y-axis and x-axis transform.
    mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2]] if len(ii) == 4 else
                                 [org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2],
                                  ii[4], ii[5]] for ii in patch_crops])

    return mirrored_patch_crops
