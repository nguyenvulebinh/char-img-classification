import tensorflow as tf
import numpy as np
from PIL import Image
import sys

output_graph = "full_model.pb"


def get_index(code):
    if code < 10:
        return code
    elif 10 <= code < 36:
        code = code - 10 + ord('A')
        return chr(code)
    else:
        code = code - 36 + ord('a')
        return chr(code)


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


# We use our "load_graph" function
graph = load_graph(output_graph)

# We can verify that we can access the list of operations in the graph
# for op in graph.get_operations():
#    print(op.name)
# prefix/Placeholder/inputs_placeholder
# ...
# prefix/Accuracy/predictions

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/input/placeholder_image:0')
y = graph.get_tensor_by_name('prefix/output/predictions:0')
prop = graph.get_tensor_by_name('prefix/input/placeholder_dropout:0')

with graph.as_default():
    input_image = np.array(Image.open(
        "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/data_1/data/4/UVNHongHaHep_B.png")).reshape(
        [1, 28, 28, 1]).astype(float)
    input_image = tf.constant(input_image, dtype=tf.float32)

# We launch a Session
with tf.Session(graph=graph) as sess:
    input_image = input_image.eval()
    y_out = sess.run(tf.argmax(y, 1), feed_dict={
        x: input_image, prop: 1.0
    })
    print(y_out, get_index(y_out))  # [[ False ]] Yay, it works!
    sess.close()
