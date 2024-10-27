const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Load YOLO model (using TensorFlow.js or another library)
async function loadModel() {
    // Replace with the path to your YOLO model
    const model = await tf.loadGraphModel('path/to/yolo/model.json');
    return model;
}

async function detectObjects(model) {
    // Capture each frame from the video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const inputTensor = tf.browser.fromPixels(video).expandDims(0);
    
    // Run YOLO on the current frame
    const predictions = await model.executeAsync(inputTensor);
    
    // Process and draw bounding boxes
    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        ctx.fillStyle = 'red';
        ctx.fillText(prediction.class, x, y > 10 ? y - 5 : y + 10);
    });

    inputTensor.dispose();
    requestAnimationFrame(() => detectObjects(model));
}

// Start video stream and run object detection
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.onloadeddata = async () => {
            const model = await loadModel();
            detectObjects(model);
        };
    })
    .catch(error => {
        console.error('Error accessing media devices.', error);
    });

    document.addEventListener('DOMContentLoaded', function() {
        const imageUrl = 'image.jpg'; 
        const imageWindow = document.getElementById('image-window');
        imageWindow.style.backgroundImage = `url(${imageUrl})`;
    });