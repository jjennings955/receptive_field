const NUM_LAYERS = 5;
const CANVAS_SIZE = [128, 128];
const SHAPE = [16,16,1];
const input = tf.input({shape:SHAPE})
layers = []

// Create a deconv net for finding the receptive field
// alternatively could use tf.grads (which would support more complex things like dilated convolutions)

for (var i = 0; i < NUM_LAYERS; i++)
{
  layers.push(tf.layers.conv2dTranspose({kernelSize:[3,3], filters: 1, useBias:false}))
}

outputs = [input, layers[0].apply(input)]

for (var i = 2; i < NUM_LAYERS; i++)
{
  outputs.push(layers[i].apply(outputs[i-1]));
}
const model = tf.model({inputs: input, outputs: outputs});

function drawReceptiveField(e)
{
pixel = [Math.floor(e.offsetY/(CANVAS_SIZE[1]/SHAPE[1])), Math.floor(e.offsetX/(CANVAS_SIZE[0]/SHAPE[0]))];
  // craft an input with a single pixel set to 1 to propagate through network
  // this will cause the output of each layer to be the receptive field for that pixel in the previous layer
  myInput = tf.oneHot(tf.tensor1d([pixel[0]*16 + pixel[1]], 'int32'),
             SHAPE[0]*SHAPE[1]*SHAPE[2]).reshape([1, ...SHAPE]).cast('float32')
  result = model.predict(myInput)
  canvases = []
  for (var i = 0; i < NUM_LAYERS; i++)
  {
    canvas = document.getElementById(`out${NUM_LAYERS - i}`)
    foo3 = result[i].notEqual(0)
    foo3.sum().print();
    tf.browser.toPixels(tf.image.resizeNearestNeighbor(foo3.squeeze(0).cast('float32'), [128,128]), canvas);
  }  
}

last_canvas = document.getElementById('out5')
last_canvas.addEventListener('mousemove', _.throttle(drawReceptiveField, 100));
last_canvas.addEventListener("touchmove", _.throttle((e) => { drawReceptiveField(e.touches[0]) }));
drawReceptiveField({ offsetX: 64, offsetY: 64});
// TODO: use svg instead of canvas so we can get the fancy looking skew transforms (doable with canvas or other libraries also)

// TODO: draw grid lines on each canvas

// TODO: construct a different model for each canvas so we can visualize the receptive field starting only at a pixel from that layer

// TODO: arbitrary