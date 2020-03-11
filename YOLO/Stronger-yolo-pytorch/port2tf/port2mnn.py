import os
from utils import *
import cv2
import torch
from port2tf.yolov3 import YOLOV3
import matplotlib.pyplot as plt
from collections import OrderedDict
import MNN
import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def collectpth(checkpoint_path):
    statedict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
    prefix2dict = {}
    for k, v in statedict.items():
        if 'num_batches_tracked' in k:
            continue
        # delete the prefix for backbone
        if 'mobilev2' in k:
            k = k.strip('mobilev2.')
        if 'backbone' in k:
            k = k.strip('backbone.')
        prefix = '.'.join(k.split('.')[:2])
        if prefix not in prefix2dict:
            prefix2dict[prefix] = [v.numpy()]
        else:
            prefix2dict[prefix].append(v.numpy())
    return prefix2dict


def freeze_graph(checkpoint_path, output_node_names, savename):
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, shape=(1, INPUTSIZE, INPUTSIZE, 3), name='input_data')
        training = tf.placeholder(dtype=tf.bool, name='training')
    prefixdict = collectpth(checkpoint_path)
    output = YOLOV3(training).build_network_dynamic(input_data, prefixdict, inputsize=INPUTSIZE,gt_per_grid=1)
    with tf.Session() as sess:
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
        with tf.gfile.GFile(savename, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def pb_test(pb_path, outnode):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
        for idx, node in enumerate(output_graph_def.node):
            if node.op == 'Conv2D' and 'explicit_paddings' in node.attr:
                del node.attr['explicit_paddings']
            if node.op == 'ResizeNearestNeighbor' and 'half_pixel_centers' in node.attr:
                del node.attr['half_pixel_centers']
        tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            img = cv2.imread('port2tf/assets/{}.jpg'.format(testimg))
            originimg = img
            orishape = img.shape
            img = preprocess(img, None, (INPUTSIZE, INPUTSIZE), False, keepratio=True)
            img = img.astype(np.float32)[np.newaxis, ...]
            inputdata = sess.graph.get_tensor_by_name("input/input_data:0")
            outbox_flag = sess.graph.get_tensor_by_name('{}:0'.format(outnode))
            outbox = sess.run(outbox_flag, feed_dict={inputdata: img})
            outbox = np.array(postprocess(outbox, INPUTSIZE, orishape[:2]))
            originimg = draw_bbox(originimg, outbox, CLASSES)
            cv2.imwrite('port2tf/assets/{}_{}.jpg'.format(testimg, savename), originimg)
            plt.imshow(originimg)
            plt.show()


def mnn_test(mnn_path):
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter(mnn_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    img = cv2.imread('port2tf/assets/{}.jpg'.format(testimg))
    originimg = img
    orishape = img.shape
    image = preprocess(img, None, (INPUTSIZE, INPUTSIZE), False, keepratio=True)

    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, INPUTSIZE, INPUTSIZE, 3), MNN.Halide_Type_Float, image,
                           MNN.Tensor_DimensionType_Tensorflow)
    # construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    output_data = np.array(output_tensor.getData())
    output_data = output_data.reshape((-1, 25))
    outbox = np.array(postprocess(output_data, INPUTSIZE, orishape[:2]))
    originimg = draw_bbox(originimg, outbox, CLASSES)
    cv2.imwrite('port2tf/assets/{}_{}mnn.jpg'.format(testimg, savename), originimg)


if __name__ == '__main__':
    INPUTSIZE = 320
    CLASSES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]
    savename = 'v3_26M'
    outnodes = "YoloV3/output/boxconcat"
    ckptpath = 'checkpoints/strongerv3_1gt/checkpoint-best.pth'
    testimg = '004650'
    freeze_graph(checkpoint_path=ckptpath, output_node_names=outnodes, savename='port2tf/assets/%s.pb' % savename)
    pb_test('port2tf/assets/%s.pb' % savename, outnodes)
    # mnn_test('port2tf/assets/%s.mnn' % savename)
    # onnx_test()
