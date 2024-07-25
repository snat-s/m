import * as tf from '@tensorflow/tfjs';

let lossChart;
const lossValues = [];

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

function parse_csv(training_data) {
    let x1 = [], x2 = [], y = [];
    for (let line of training_data.split("\n")) {
        const [x1_val, x2_val, y_val] = line.split(",");
        x1.push(parseInt(x1_val));
        x2.push(parseInt(x2_val));
        y.push(parseInt(y_val));
    }
    console.log("Parsed csv");
    //console.log(x1, x2, y);
    return [x1, x2, y];
}

const response = await fetch("./train.csv");
const training_data = await response.text();
const [x1, x2, y] = parse_csv(training_data);

const xsData = x1.map((val, i) => [val, x2[i]]);
const xs = tf.tensor2d(xsData, [x1.length, 2]);
const ys = tf.tensor2d(y, [y.length, 1]);


function runTraining() {
    generateTrainingSetImage(xs);
    initLossChart();

     model.fit(xs, ys, {
        epochs: 20,
        callbacks: {
            onEpochEnd: onEpochEnd
        }
    });

    console.log("Training complete");
}

function generateTrainingSetImage(xs) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const width = 32;
    const height = 22;
    canvas.width = width;
    canvas.height = height;
    const imageData = ctx.createImageData(width, height);
  
    const xsData = xs.dataSync();
    for (let i = 0; i < xsData.length; i += 2) {
      const x1 = xsData[i];
      const x2 = xsData[i + 1];
      const index = (i / 2) * 4;
  
      imageData.data[index] = x1;     // R (represents x1)
      imageData.data[index + 1] = x2; // G (represents x2)
      imageData.data[index + 2] = 0;  // B (unused)
      imageData.data[index + 3] = 255; // A (fully opaque)
    }
  
    ctx.putImageData(imageData, 0, 0);
  
    // Display the image
    const trainingSetImg = document.getElementById('trainingSet');
    trainingSetImg.src = canvas.toDataURL();
  }

runTraining()

function initLossChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function updateLossChart(epoch, loss) {
    if (lossChart && lossChart.data) {
        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update();
    }
}

function onEpochEnd(epoch, logs) {
    console.log(`Epoch ${epoch + 1} - Loss: ${logs.loss.toFixed(4)}`);
    updateLossChart(epoch + 1, logs.loss);
    
    const prediction = model.predict(xs);
    //console.log((xs == ys).sum());
    generateImage(prediction);
    prediction.dispose();
}

function generateImage(prediction) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const width = 32;
    const height = 22;
    canvas.width = width;
    canvas.height = height;
    
    const imageData = ctx.createImageData(width, height);
    const predictionData = prediction.dataSync();
    
    for (let i = 0; i < predictionData.length; i++) {
        const value = predictionData[i] >= 0.5 ? 0 : 255; // Black if >= 0.5, White if < 0.5
        const index = i * 4;
        imageData.data[index] = value;     // R
        imageData.data[index + 1] = value; // G
        imageData.data[index + 2] = value; // B
        imageData.data[index + 3] = 255;   // A (fully opaque)
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Display the image
    const outputImg = document.getElementById('output');
    outputImg.src = canvas.toDataURL();
}

// Wait for the DOM to be fully loaded before running the script
document.addEventListener('DOMContentLoaded', (event) => {
    runTraining();
});