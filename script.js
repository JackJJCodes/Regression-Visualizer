let xVals = [];
let yVals = [];

let m, c;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(400, 400);

    m = tf.variable(tf.scalar(random(1)));
    c = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
     return pred.sub(labels).square().mean();
}

function predict(x) {
    const xs = tf.tensor1d(x);
    // y = mx + c;
    const ys = xs.mul(m).add(c);

    return ys;
}

function mousePressed() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    xVals.push(x);
    yVals.push(y);
}

function draw() {

    if (xVals.length > 0) {
        const ys = tf.tensor1d(yVals);
        optimizer.minimize(() => loss(predict(xVals), ys));
    }

    background(0);
    stroke(255);
    strokeWeight(8);
    for(let i = 0; i < xVals.length; i++) {
        let px = map(xVals[i], 0, 1, 0, width);
        let py = map(yVals[i], 0, 1, height, 0);
        point(px, py);
    }

    const lineX = [0, 1];
    const ys = tf.tidy (() => predict(lineX));
    let lineY = ys.dataSync();
    ys.dispose();
    // ys.print();

    let x1 = map(lineX[0], 0, 1, 0, width);
    let x2 = map(lineX[1], 0, 1, 0, width);
    // console.log(lineY);
    let y1 = map(lineY[0], 0, 1, height, 0);
    let y2 = map(lineY[1], 0, 1, height, 0);
    line(x1, y1, x2, y2);
}