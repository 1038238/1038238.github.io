// Largely based on https://github.com/tensorflow/tfjs-models/blob/master/mobilenet/src/index.ts
const webcamElement = document.getElementById('webcam');
let model;
let loading = true;
let working = false;
let selected;
let idx_new_img = 0

const IMAGE_SIZE = 224;  
const CLASSES = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
const MODEL_LOCATION = 'mobilenetjs_quantised/model.json'

var options = {
     theme:"sk-cube-grid",
     message:"Preparing MobileNet for your browser...",
};

HoldOn.open(options);

// Try to do everything asynchronously to avoid blocking the UI thread for too long...
async function warmupModel() {
    console.log("Warming up model...")
    result = model.predict(tf.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE]));
    result.data().then(d => {
        result.dispose();
        HoldOn.close();
        console.log("Finished warming up model.")
    });
}

async function loadModel() {
    
    function updateProgress(p) {
        console.log("Loading model: "+p+" / 1");
    }

    // Load the model
    console.log('Loading MobileNetV2..');
    //model = await tf.loadLayersModel(MODEL_LOCATION, { onProgress: updateProgress });
    tf.loadLayersModel(MODEL_LOCATION, { onProgress: updateProgress }).then(m => {
        model = m;
        console.log('Successfully loaded model.');
        document.getElementById("holdon-message").innerHTML = "Done";
        warmupModel();
    });
    
}

async function classifyImage(imgElement) {
    if(working) return;
    if(!selectImage(imgElement)) return;
    working = true;
    
    let input = await preprocess(imgElement);
    console.log("Classifying image...");
    var t0 = performance.now();
    const classProbabilities = await classify(model, input);
    var t1 = performance.now();
    console.log("Prediction took " + (t1 - t0) + " milliseconds.");
    console.log(CLASSES[classProbabilities[0].cls]);
    displayResults(classProbabilities, t1-t0);
    working = false;
}

// Prepare the image for the network (normalize, resize, 
async function preprocess(imgElement) {
    var img = tf.browser.fromPixels(imgElement);
    
    // Normalize: (img - mean) / std
    // torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
    //                                  [0.229, 0.224, 0.225])
    const normalized = img.toFloat().div(255)
                        .sub(tf.tensor([0.485, 0.456, 0.406]))
                        .div(tf.tensor([0.229, 0.224, 0.225]));

    let resized = normalized;
    if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(
            normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners);
    } 
    
    img = tf.transpose(resized, [2,0,1]) // swap dimensions for PyTorch channels first (w,h,c) -> (c,w,h)
    img = img.expandDims()               // insert a dimension to make it a batch (1,c,w,h)
    return img;  
}

async function classify(model, image) {
    const r = model.predict(image);
    const logits = tf.tensor1d(await r.data())
    const softmax = logits.softmax();
    const values = await softmax.data();
    console.log(values)
    softmax.dispose();
    logits.dispose();
    
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
        valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
        return b.value - a.value;
    });
    topK = 3
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
        topkValues[i] = valuesAndIndices[i].value;
        topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
        topClassesAndProbs.push({
            cls: topkIndices[i],
            prob: topkValues[i]
        });
    }
    return topClassesAndProbs;
}

function selectImage(imgElement) {
    if(selected) {
        if(selected === imgElement) return false;
        selected.classList.remove("selected")
    }
    selected = imgElement;
    selected.classList.add("selected")
    return true;
}

function displayResults(classProbabilities, inferenceTime) {
    document.getElementById("inference-time").scrollIntoView();
    console.log(classProbabilities)
    $( "#prompt" ).hide();//remove()
    $( "#inference-time" ).text("Inference Time - "+Math.round(inferenceTime)+"ms");
    for(var i = 0; i < 3; i++) {
        let parent = $( "#p"+i )
        parent.find( ".label" ).first().text(CLASSES[classProbabilities[i].cls])
        parent.find( ".probability" ).first().text((Math.round(classProbabilities[i].prob*10000)/100).toFixed(2)+"%")
    }
}

function newImage(){
	
	let rand = false;
	
	var num_img = 58
	if(idx_new_img + 1 <= num_img && !rand){
    	idx_new_img = idx_new_img + 1
    }else{
    	idx_new_img = Math.floor((Math.random()*num_img)+1);
    	rand = true;
    }
    
    
    console.log("Image " + idx_new_img)
	
	//var randomNum = Math.floor((Math.random()*N)+1);
	document.getElementById("img").src = "images/image_" + idx_new_img + ".jpg";
	selected = null;
	$( "#prompt" ).show()
	$( "#inference-time" ).text("");
	for(var i = 0; i < 3; i++) {
        let parent = $( "#p"+i )
        parent.find( ".label" ).first().text("")
        parent.find( ".probability" ).first().text("")
    }
    
    
	

}

window.onload = function(e) {
    console.log('DOM loaded');
    loadModel();
    console.log("Running Tensorflow.js on "+tf.getBackend());
}
