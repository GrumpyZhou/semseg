"""Compact interfaces lib for a neural network including:
-- Interfaces to define a nn layer e.g conv, pooling, relu, fcn, dropout etc
-- Interfaces for variable initialization
-- Interfaces for network data post-processing e.g logging, visualizing and so on
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import nn
import math
import data_utils as dt
from scipy import misc

DATA_DIR = 'data'

class InstanceSensitiveFCN8s:

    def __init__(self, data_path=None):
        # Load pretrained weight
        data_dict = dt.load_weight(data_path)
        self.data_dict = data_dict

        # used to save trained weights
        self.var_dict = {}


    def _build_model(self, image, is_train=False, save_var=False):

        model = {}
        # During training, use pretrained vgg16 weight
        feed_dict = self.data_dict

        if save_var:
            # During training, weights are saved to var_dict
            var_dict = self.var_dict
        else:
            # During inference or validation, no need to save weights
            var_dict = None

        # Base channel: build fcn8s and produce feature maps with shape[H, W, 512]
        model['conv1_1'] = nn.conv_layer(image, feed_dict, "conv1_1", var_dict=var_dict)
        model['conv1_2'] = nn.conv_layer(model['conv1_1'], feed_dict, "conv1_2", var_dict=var_dict)
        model['pool1'] = nn.max_pool_layer(model['conv1_2'], "pool1")

        model['conv2_1'] = nn.conv_layer(model['pool1'], feed_dict, "conv2_1", var_dict=var_dict)
        model['conv2_2'] = nn.conv_layer(model['conv2_1'], feed_dict, "conv2_2", var_dict=var_dict)
        model['pool2'] = nn.max_pool_layer(model['conv2_2'], "pool2")

        model['conv3_1'] = nn.conv_layer(model['pool2'], feed_dict, "conv3_1", var_dict=var_dict)
        model['conv3_2'] = nn.conv_layer(model['conv3_1'], feed_dict, "conv3_2", var_dict=var_dict)
        model['conv3_3'] = nn.conv_layer(model['conv3_2'], feed_dict, "conv3_3", var_dict=var_dict)
        model['pool3'] = nn.max_pool_layer(model['conv3_3'], "pool3")

        model['conv4_1'] = nn.conv_layer(model['pool3'], feed_dict, "conv4_1", var_dict=var_dict)
        model['conv4_2'] = nn.conv_layer(model['conv4_1'], feed_dict, "conv4_2", var_dict=var_dict)
        model['conv4_3'] = nn.conv_layer(model['conv4_2'], feed_dict, "conv4_3", var_dict=var_dict)
        model['pool4'] = nn.max_pool_layer(model['conv4_3'], "pool4")


        model['conv5_1'] = nn.conv_layer(model['pool4'], feed_dict, "conv5_1", var_dict=var_dict)
        model['conv5_2'] = nn.conv_layer(model['conv5_1'], feed_dict, "conv5_2", var_dict=var_dict)
        model['conv5_3'] = nn.conv_layer(model['conv5_2'], feed_dict, "conv5_3", var_dict=var_dict)
        model['pool5'] = nn.max_pool_layer(model['conv5_3'], "pool5")

        model['conv6_1'] = nn.conv_layer(model['pool5'], feed_dict, "conv6_1",
                                         shape=[3, 3, 512, 512], dropout=is_train,
                                         keep_prob=0.5, var_dict=var_dict)

        model['conv6_2'] = nn.conv_layer(model['conv6_1'], feed_dict, "conv6_2",
                                         shape=[3, 3, 512, 512], dropout=is_train,
                                         keep_prob=0.5, var_dict=var_dict)

        model['conv6_3'] = nn.conv_layer(model['conv6_2'], feed_dict, "conv6_3",
                                         shape=[3, 3, 512, 4096], dropout=is_train,
                                         keep_prob=0.5, var_dict=var_dict)

        model['conv7'] = nn.conv_layer(model['conv6_3'], feed_dict, "conv7",
                                       shape=[1, 1, 4096, 4096], dropout=is_train,
                                       keep_prob=0.5, var_dict=var_dict)

        # Skip feature fusion
        model['score_fr'] = nn.conv_layer(model['conv7'], feed_dict, "score_fr_inst",
                                          shape=[1, 1, 4096, 512], relu=False,
                                          dropout=False, var_dict=var_dict)

        # Upsample: score_fr*2
        upscore_fr_2s = nn.upscore_layer(model['score_fr'], feed_dict, "upscore_fr_2s_inst",
                                       tf.shape(model['pool4']), num_class=512,
                                       ksize=4, stride=2, var_dict=var_dict)
        # Fuse upscore_fr_2s + score_pool4
        in_features = model['pool4'].get_shape()[3].value
        score_pool4 = nn.conv_layer(model['pool4'], feed_dict, "score_pool4_inst",
                                    shape=[1, 1, in_features, 512],
                                    relu=False, dropout=False, var_dict=var_dict)

        fuse_pool4 = tf.add(upscore_fr_2s, score_pool4)


        # Upsample fuse_pool4*2
        upscore_pool4_2s = nn.upscore_layer(fuse_pool4, feed_dict, "upscore_pool4_2s_inst",
                                            tf.shape(model['pool3']), num_class=512,
                                            ksize=4, stride=2, var_dict=var_dict)

        # Fuse upscore_pool4_2s + score_pool3
        in_features = model['pool3'].get_shape()[3].value
        score_pool3 = nn.conv_layer(model['pool3'], feed_dict, "score_pool3_inst",
                                    shape=[1, 1, in_features, 512],
                                    relu=False, dropout=False, var_dict=var_dict)

        model['score_out'] = tf.add(upscore_pool4_2s, score_pool3)


        # Instance assembling score
        model['inst_conv1'] = nn.conv_layer(model['score_out'], feed_dict, "inst_conv1",
                                            shape=[1, 1, 512, 512],relu=False,
                                            dropout=False, var_dict=var_dict)

        model['inst_score'] = nn.conv_layer(model['inst_conv1'], feed_dict, "inst_conv2",
                                            shape=[3, 3, 512, 9],relu=False,
                                            dropout=False, var_dict=var_dict)

        # Objectness score
        model['obj_conv1'] = nn.conv_layer(model['score_out'], feed_dict, "obj_conv1",
                                            shape=[3, 3, 512, 512],relu=False,
                                            dropout=False, var_dict=var_dict)

        model['obj_score'] = nn.conv_layer(model['obj_conv1'], feed_dict, "obj_conv2",
                                            shape=[1, 1, 512, 1],relu=False,
                                            dropout=False, var_dict=var_dict)

        """
        # Upsample to original size *8 # Or we have to do it by class
        model['upmask'] = nn.upscore_layer(score_out, feed_dict,
                                  "upmask", tf.shape(image), self.num_pred_class * max_instance,
                                  ksize=16, stride=8, var_dict=var_dict)
        """


        print('Instance-sensitive-fcn8s model is builded successfully!')
        print('Model: %s' % str(model.keys()))
        return model


    def train(self, image, gt_mask, gt_box, learning_rate=1e-6, num_box = 256, save_var=True):
        '''
        Input
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32
        gt_masks: stacked instance_masks, shape=[1, h, w, num_gt_class], tf.int32
        '''
        # Build basic model
        model = self._build_model(image, is_train=True, save_var=save_var)

        # Assemble instance proposals during training with given location and size of bounding boxs
        inst_score = model['inst_score']
        obj_score = model['obj_score']
        loss = 0

        for k in range(num_box):
            # box location and size
            x = gt_box[k][0][0]
            y = gt_box[k][0][1]
            w = gt_box[k][1][0]
            h = gt_box[k][1][1]

            gt_flag = gt_box[k][2][0]
            obj_id = gt_box[k][2][1]
            cond = tf.equal(1, gt_flag)

            # Slice out instance_gt_score, ignore irrelevant instances
            instance_gt_ = tf.slice(gt_mask, [0, x, y, 0], [1, w, h,1])
            indices = tf.where(tf.equal(instance_gt_, obj_id))
            sparse_val = tf.constant(1, dtype=tf.float32)
            instance_gt_shape = tf.to_int64([1, w, h, 1])
            instance_gt_shape = tf.pack(instance_gt_shape)
            instance_gt_score = tf.sparse_to_dense(indices, instance_gt_shape, sparse_val)

            # Calculate weight
            inst_pixel = tf.cast(tf.shape(indices)[0], tf.float32)
            total_pixel = tf.cast(w * h, tf.float32)
            weight = (total_pixel - inst_pixel) / inst_pixel

            # Downscale the bounding box
            w_s =  tf.cast(tf.floor(w / 8), tf.int32)
            h_s =  tf.cast(tf.floor(h / 8), tf.int32)
            x_s = tf.cast(tf.floor(x / 8), tf.int32)
            y_s = tf.cast(tf.floor(y / 8), tf.int32)
            
            # Calculate instance loss (only for positive samples) and objectness loss
            inst_loss = tf.cond(cond, lambda: self.get_inst_loss(inst_score, instance_gt_score, weight, x_s, y_s, w_s, h_s), lambda: tf.constant(0, tf.float32))
            obj_loss = self.get_obj_loss(obj_score, instance_gt_score, weight, x_s, y_s, w_s, h_s)
            loss += obj_loss + inst_loss
            
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step, loss

    def get_obj_loss(self, obj_score, obj_gt, weight, x, y, w, h):
        # Generate corresponding objectness score
        objectness = tf.slice(obj_score, [0, x, y, 0], [1, w, h, 1])

        # Upsample instance proposal to original size *8
        objectness = tf.image.resize_bilinear(objectness, [w*8, h*8])
        
        # Logistic regression to calculate loss weighted cross-entropy)
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=objectness, targets=obj_gt, pos_weight=weight))
        return loss

    def get_inst_loss(self, inst_score, inst_gt, weight, x, y, w, h):
        # Generate instance proposal
        inst_shape = tf.shape(inst_score)
        sub_score = tf.slice(inst_score, [0, x, y, 0], [1, w, h, inst_shape[3]])
        instance = self.assemble(w, h, sub_score)

        # Upsample instance proposal to original size *8
        instance = tf.image.resize_bilinear(instance, [w*8, h*8])
        
        # Logistic regression to calculate loss (weighted cross-entropy)
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=instance, targets=inst_gt,  pos_weight=weight))
        return loss

    def assemble(self, w, h, score, k=3):
        """
        Assemble  k*k parts across last dimension to make one instance proposal
        Return a tensor with the shape [1, w, h, 1] 
        """
        dx = tf.floor(w / k)
        dy = tf.floor(h / k)
        dx = tf.cast(dx, tf.int32)
        dy = tf.cast(dy, tf.int32)
        parts = []
        for i in range(k):
            for j in range(k):
                c = tf.constant(i*k+j, tf.int32)
                parts.append(tf.slice(score, [0, i*dx, j*dy, c], [1, dx, dy, 1]))

        # concat along x first, then concat along y
        concated_x = []
        for i in range(k*k):
            parts[i] = tf.reshape(parts[i], [dx,dy])
        for i in range(k):
            temp_x = tf.concat(1,[parts[i*3+0], parts[i*3+1], parts[i*3+2]])
            concated_x.append(temp_x)
        instance = tf.concat(0,concated_x)

        instance = tf.reshape(instance, [1, dx*k, dy*k, 1])
        return instance
    

    def inference(self, image):
        # Build model
        model = self._build_model(image, is_train=False)
        obj_score = model['obj_score']
        inst_score = model['inst_score']
        return tf.squeeze(obj_score),  tf.squeeze(inst_score)

    def dense_assemble(self, score_map, feature_map, sz=21, stride=8, w=128, h=256, threshold=0.8):
        """
        Assemble instance proposal for inference, implemented with numpy.
        """
        # Sliding over score map
        object_pos = []
        ix = int((w - sz) / stride)
        iy = int((h - sz) / stride)
        pos_x = 0
        for i in range(ix):
            pos_y = 0
            for j in range(iy):
                # calculate objectness score
                objectness = score_map[pos_x : (pos_x + sz), pos_y : (pos_y + sz)]
                score = np.mean(objectness)
                object_pos.append([pos_x, pos_y, pos_x + sz, pos_y + sz, score])

                pos_y += stride + sz
                if ((pos_y + sz) >= h):
                    break

            pos_x += stride + sz
            if ((pos_x + sz) >= w):
                break

        # Convert list to np array
        object_pos = np.array(object_pos)
        # Get bounding boxes of instances with nonmaximum suppression
        picked_pos = self.non_max_suppression_fast(object_pos, area=w*h, threshold=threshold)
        print('object_pos %s picked_pos %s' % (object_pos.shape,picked_pos.shape))

        # Make instance proposals and fuse all instances into one image
        instances = []
        prediction = np.zeros((w * 8, h * 8))
        part_num = 3
        dx = int(sz / part_num)
        dy = int(sz / part_num)
        instances = []
        for k in range(picked_pos.shape[0]):
            pos = picked_pos[k]
            sub_feature = feature_map[pos[0] : pos[2], pos[1] : pos[3], :]     
            proposal = []
            for i in range(part_num):
                row = []
                for j in range(part_num):
                    c = i * part_num + j
                    row.append(sub_feature[i * dx : (i + 1) * dx, j * dy : (j + 1) * dy, c])
                row = np.hstack(row)
                proposal.append(row)        
            proposal = np.vstack(proposal)

            # upsample instance to 8*wind_sz
            instance = misc.imresize(proposal, (sz * 8, sz * 8), interp='bilinear')
            prediction[pos[0] * 8 : pos[2] * 8, pos[1] * 8 : pos[3] * 8] += instance 
            #instances.append(instance)
        return prediction #instances, 8 * picked_pos


    def non_max_suppression_fast(self, boxes, area=128*256, threshold=0.8):
    
        """
        Modified non-maximum suppresion based on the implementation by Adrian Rosebrock
        Source: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/ 
        """

	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes and corresponding score
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
        score = boxes[:,4]
        
        # Sort with score, the higher the score, the more confident it is an instance bounding box
	idxs = np.argsort(score)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > threshold)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int") 
