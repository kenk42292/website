/* global $, document, MathJax */

/* net parameters */
function changeNumLayers() { //eslint-disable-line no-unused-vars
    var x = document.getElementById("num-layers").value;
    var net1 = document.getElementById("net-1");
    var net2 = document.getElementById("net-2");
    if (x === "1") {
        net1.style.display="block";
        net2.style.display="none";
    } else if (x === "2") {
        net1.style.display="none";
        net2.style.display="block";
    }
}

function changeNonLinearity() { //eslint-disable-line no-unused-vars
    var x = document.getElementById("nonlin-type").value;
    var nonLins = [document.getElementById("net-1-nonlin-1"),
                   document.getElementById("net-2-nonlin-1"),
                   document.getElementById("net-2-nonlin-2")];
    for (var i=0; i<nonLins.length; i++) {
        if (x === "linear") {
            nonLins[i].src = "images/linear.png";
        } else if (x === "sigmoid") {
            nonLins[i].src = "images/sigmoid.png";
        } else if (x === "softplus") {
            nonLins[i].src = "images/softplus.png";
        }
    }
}

function changeMode() { //eslint-disable-line no-unused-vars
    var mode = document.getElementById("mode").value;
    if (mode === "training") {
        document.getElementById("adjust-button").style.display="block";
        document.getElementById("train-button").style.display="block";
        document.getElementById("validate-button").style.display="none";
        document.getElementById("accuracy").style.display="none";
    } else {
        document.getElementById("adjust-button").style.display="none";
        document.getElementById("train-button").style.display="none";
        document.getElementById("validate-button").style.display="block";
        document.getElementById("accuracy").style.display="block";
    }
}

/* control buttons */
function init() { //eslint-disable-line no-unused-vars
    // intialize rmsProp learning rate params
    net1_avg_grad_sq_w1 = null;
    net1_avg_grad_sq_b1 = null;
    net2_avg_grad_sq_w1 = null;
    net2_avg_grad_sq_b1 = null;
    net2_avg_grad_sq_w2 = null;
    net2_avg_grad_sq_b2 = null;

    // initialize nets
    if (isNet1()) {
        var net1layer1Jax = MathJax.Hub.getAllJax("net-1-layer-1");
        net1mat1 = randMatrix(3, 4, 2);
        MathJax.Hub.Queue(["Text", net1layer1Jax[0], "* " + matrix2Str(net1mat1)]);
        net1bias1 = randMatrix(3, 1, 2);
        MathJax.Hub.Queue(["Text", net1layer1Jax[1], "+ " + matrix2Str(net1bias1)]);
    }
    if (isNet2()) {
        var net2layer1Jax = MathJax.Hub.getAllJax("net-2-layer-1");
        net2mat1 = randMatrix(3, 4, 2);
        MathJax.Hub.Queue(["Text", net2layer1Jax[0], "* " + matrix2Str(net2mat1)]);
        net2bias1 = randMatrix(3, 1, 2);
        MathJax.Hub.Queue(["Text", net2layer1Jax[1], "+ " + matrix2Str(net2bias1)]);

        var net2layer2Jax = MathJax.Hub.getAllJax("net-2-layer-2");
        net2mat2 = randMatrix(3, 3, 2);
        MathJax.Hub.Queue(["Text", net2layer2Jax[0], "* " + matrix2Str(net2mat2)]);
        net2bias2 = randMatrix(3, 1, 2);
        MathJax.Hub.Queue(["Text", net2layer2Jax[1], "+ " + matrix2Str(net2bias2)]);

    }
}

function nextSample(display) { //eslint-disable-line no-unused-vars
    var mode = document.getElementById("mode").value;
    var sample;
    if (mode === "training") {
        sample = trainingData[Math.floor(Math.random()*trainingData.length)];
    } else {
        sample = validationData[Math.floor(Math.random()*validationData.length)];
    }

    x = [];
    for (var i=0; i<4; i++) { // column vector as 2D matrix
        x.push([parseFloat(sample[i]),]);
    }
    ystar = [[0.0,], [0.0,], [0.0,]]; // column vector as 2D matrix
    ystar[sample[4]][0] = 1.0;

    if (display) {
        /* Net-1 sample */
        var net1mathJax = MathJax.Hub.getAllJax("net-1");
        MathJax.Hub.Queue(["Text", net1mathJax[0], matrix2Str(x)]);
        MathJax.Hub.Queue(["Text", net1mathJax[net1mathJax.length-2], matrix2Str(ystar)]);

        /* Net-2 sample */
        var net2mathJax = MathJax.Hub.getAllJax("net-2");
        MathJax.Hub.Queue(["Text", net2mathJax[0], matrix2Str(x)]);
        MathJax.Hub.Queue(["Text", net2mathJax[net2mathJax.length-2], matrix2Str(ystar)]);
    }
}

function predict(display) { //eslint-disable-line no-unused-vars
    if (isNet1()) {
        z1 = elemWiseAdd(multMat(net1mat1, x), net1bias1);
        y1 = applyNonlin(z1);
        yhat = softmax(y1);
        if (display) {
            var net1ystarJax = MathJax.Hub.getAllJax("net-1-y");
            MathJax.Hub.Queue(["Text", net1ystarJax[0], matrix2Str(yhat)]);
        }
    } else if (isNet2()) {
        z1 = elemWiseAdd(multMat(net2mat1, x), net2bias1);
        y1 = applyNonlin(z1);

        z2 = elemWiseAdd(multMat(net2mat2, y1), net2bias2);
        y2 = applyNonlin(z2);

        yhat = softmax(y2);
        if (display) {
            var net2ystarJax = MathJax.Hub.getAllJax("net-2-y");
            MathJax.Hub.Queue(["Text", net2ystarJax[0], matrix2Str(yhat)]);
        }
    }
}

function backprop(display) { //eslint-disable-line no-unused-vars
    var dLdy, dLdz, dydz, dLdw, dLdb;
    var mat_vals, bias_vals;
    if (isNet1()) {
        dLdy = elemWiseSubtract(yhat, ystar); // cross entropy loss with softmax
        dydz = applyNonlinPrime(z1);
        dLdz = elemWiseMult(dLdy, dydz);
        dLdw = multMat(dLdz, transpose(x));
        dLdb = dLdz;
        mat_vals = rmsProp(net1mat1, net1_avg_grad_sq_w1, dLdw);
        net1_avg_grad_sq_w1 = mat_vals[0];
        net1mat1 = mat_vals[1];
        bias_vals = rmsProp(net1bias1, net1_avg_grad_sq_b1, dLdb);
        net1_avg_grad_sq_b1 = bias_vals[0];
        net1bias1 = bias_vals[1];
        if (display) {
            var net1layer1Jax = MathJax.Hub.getAllJax("net-1-layer-1");
            MathJax.Hub.Queue(["Text", net1layer1Jax[0], matrix2Str(net1mat1)]);
            MathJax.Hub.Queue(["Text", net1layer1Jax[1], matrix2Str(net1bias1)]);
        }
    } else if (isNet2()) {
        dLdy = elemWiseSubtract(yhat, ystar); // dL/dy2
        dydz = applyNonlinPrime(z2); // dy2/dz2
        dLdz = elemWiseMult(dLdy, dydz); // dL/dz2 = dL/dy2 * dy2/dz2
        dLdw = multMat(dLdz, transpose(y1)); // dL/dw2 = dL/dz2 * dz2/dw2
        dLdb = dLdz;
        mat_vals = rmsProp(net2mat2, net2_avg_grad_sq_w2, dLdw);
        net2_avg_grad_sq_w2 = mat_vals[0];
        var net2mat2_copy = copy_mat(net2mat2);
        net2mat2 = mat_vals[1];
        bias_vals = rmsProp(net2bias2, net2_avg_grad_sq_b2, dLdb);
        net2_avg_grad_sq_b2 = bias_vals[0];
        net2bias2 = bias_vals[1];
        if (display) {
            var net2layer2Jax = MathJax.Hub.getAllJax("net-2-layer-2");
            MathJax.Hub.Queue(["Text", net2layer2Jax[0], matrix2Str(net2mat2)]);
            MathJax.Hub.Queue(["Text", net2layer2Jax[1], matrix2Str(net2bias2)]);
        }

        dLdy = multMat(transpose(net2mat2_copy), dLdz); // dL/dy1 = dL/dz2 * dz2/dy1
        dydz = applyNonlinPrime(z1); // dy1/dz1
        dLdz = elemWiseMult(dLdy, dydz); // dL/dz1 = dL/dy1 * dy1/dz1
        dLdw = multMat(dLdz, transpose(x)); // dL/dw1 = dL/dz1 * dz1/dw1
        dLdb = dLdz;
        mat_vals = rmsProp(net2mat1, net2_avg_grad_sq_w1, dLdw);
        net2_avg_grad_sq_w1 = mat_vals[0];
        net2mat1 = mat_vals[1];
        bias_vals = rmsProp(net2bias1, net2_avg_grad_sq_b1, dLdb);
        net2_avg_grad_sq_b1 = bias_vals[0];
        net2bias1 = bias_vals[1];
        if (display) {
            var net2layer1Jax = MathJax.Hub.getAllJax("net-2-layer-1");
            MathJax.Hub.Queue(["Text", net2layer1Jax[0], matrix2Str(net2mat1)]);
            MathJax.Hub.Queue(["Text", net2layer1Jax[1], matrix2Str(net2bias1)]);
        }
    }
}

function train(iters) { //eslint-disable-line no-unused-vars
    for (var i=0; i<iters; i++) {
        nextSample(false);
        predict(false);
        backprop(false);
    }
    nextSample(true);
    predict(true);
    backprop(true);
}

function validate() { //eslint-disable-line no-unused-vars
    var numCorrect = 0.0;
    var sample;
    for (var i=0; i<validationData.length; i++) {
        sample = validationData[i];
        x = [];
        for (var j=0; j<4; j++) { // column vector as 2D matrix
            x.push([parseFloat(sample[j]),]);
        }
        ystar = [[0.0,], [0.0,], [0.0,]]; // column vector as 2D matrix
        ystar[sample[4]][0] = 1.0;
        
        predict(false);
        if (argmax(ystar) === argmax(yhat)) {
            numCorrect += 1.0;
        }
    }
    var accuracy = numCorrect / validationData.length;
    document.getElementById("accuracy").innerHTML = "Accuracy: " + accuracy;
}

/* helper functions */
function isNet1() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value === "1";
}

function isNet2() { //eslint-disable-line no-unused-vars
    return document.getElementById("num-layers").value === "2";
}

function applyNonlin(mat) { //eslint-disable-line no-unused-vars
    var type = document.getElementById("nonlin-type").value;
    var result = [];
    for (var i=0; i<mat.length; i++) {
        var r = [];
        for (var j=0; j<mat[i].length; j++) {
            if (type === "linear") {
                r.push(mat[i][j]);
            } else if (type === "sigmoid") {
                r.push(sigmoid(mat[i][j]));
            } else if (type === "softplus") {
                r.push(softplus(mat[i][j]));
            }
        }
        result.push(r); 
    }
    return result;
}

function applyNonlinPrime(z) { //eslint-disable-line no-unused-vars
    var type = document.getElementById("nonlin-type").value;
    var result = [];
    for (var i=0; i<z.length; i++) {
        var r = [];
        for (var j=0; j<z[i].length; j++) {
            if (type === "linear") {
                r.push(1.0);
            } else if (type === "sigmoid") {
                r.push(sigmoid_prime(z[i][j]));
            } else if (type === "softplus") {
                r.push(sigmoid(z[i][j]));
            }
        }
        result.push(r);
    }
    return result;
}

function sigmoid(x) {
    return 1.0/(1.0+Math.exp(-x));
}

function softplus(x) {
    return Math.log(1.0 + Math.exp(x));
}

// vector: column vector, as 2D matrix
function softmax(vector) {
    var r = [];
    var total = 0.0;
    for (var i=0; i<vector.length; i++) {
        var u = Math.exp(vector[i][0]);
        r.push(u);
        total += u;
    }

    var result = [];
    for (i=0; i<r.length; i++) {
        result.push([r[i]/total]);
    }
    return result;
}

function sigmoid_prime(x) {
    return sigmoid(x)*(1.0-sigmoid(x));
}

function argmax(vector) {
    var maxIndex = 0;
    var maxVal = null;
    for (var i=0; i<vector.length; i++) {
        if (maxVal == null || vector[i][0] > maxVal) {
            maxIndex = i;
            maxVal = vector[i][0];
        }
    }
    return maxIndex;
}

function matrix2Str(matrix) {
    var s = "\\begin{bmatrix} ";
    for (var i=0; i<matrix.length; i++) {
        var delim = "";
        for (var j=0; j<matrix[i].length; j++) {
            s = s.concat(delim);
            s = s.concat(matrix[i][j].toFixed(2));
            delim = " & ";
        }
        s = s.concat(" \\\\ ");
    }
    s = s.concat(" \\end{bmatrix}");
    return s;
}

function randMatrix(height, width, scale) {
    var mat = [];
    for (var i=0; i<height; i++) {
        var row = [];
        for (var j=0; j<width; j++) {
            row.push((Math.random()-0.5)*scale);
        }
        mat.push(row);
    }
    return mat;
}

function multMat(mat1, mat2) {
    var result = [];
    for (var i=0; i<mat1.length; i++) { // i: rows of mat1
        var row = [];
        for (var k=0; k<mat2[0].length; k++) { // k: columns of mat2
            var elem = 0;
            for (var j=0; j<mat1[0].length; j++) { // j: columns of mat1, rows of mat2
                elem += mat1[i][j]*mat2[j][k];
            }
            row.push(elem);
        }
        result.push(row);
    }
    return result;
}

function elemWiseMult(mat1, mat2) {
    var result = [];
    for (var i=0; i<mat1.length; i++) {
        var r = [];
        for (var j=0; j<mat1[i].length; j++) {
            r.push(mat1[i][j]*mat2[i][j]);
        }
        result.push(r);
    }
    return result;
}

function elemWiseSubtract(mat1, mat2) {
    var result = [];
    for (var i=0; i<mat1.length; i++) {
        var r = [];
        for (var j=0; j<mat1[i].length; j++) {
            r.push(mat1[i][j]-mat2[i][j]);
        }
        result.push(r);
    }
    return result;
}

function elemWiseAdd(mat1, mat2) {
    var result = [];
    for (var i=0; i<mat1.length; i++) {
        var r = [];
        for (var j=0; j<mat1[i].length; j++) {
            r.push(mat1[i][j]+mat2[i][j]);
        }
        result.push(r);
    }
    return result;
}

// current_mat: current weights, or biases
// avg_sq_mat: averged squared matrix
// gradient_mat: dLdw, or dLdb
// USE: net1_avg_grad_sq_w1, net1mat1 = rmsProp(net1mat1, net1_avg_grad_sq_w1, dLdw)
function rmsProp(current_mat, avg_sq_mat, gradient_mat) {
    var i, j, row;

    // avg_sq_mat update
    var result_avg_sq_mat = [];
    if (avg_sq_mat == null) {
        result_avg_sq_mat = elemWiseMult(gradient_mat, gradient_mat);
    } else {
        for (i=0; i<avg_sq_mat.length; i++) {
            row = [];
            for (j=0; j<avg_sq_mat[i].length; j++) {
                row.push(gamma*avg_sq_mat[i][j] + (1.0-gamma)*gradient_mat[i][j]*gradient_mat[i][j]);
            }
            result_avg_sq_mat.push(row);
        }
    }

    // result calculation
    var result = [];
    for (i=0; i<current_mat.length; i++) {
        var r = [];
        for (j=0; j<current_mat[i].length; j++) {
            r.push(current_mat[i][j] - eta*gradient_mat[i][j]/(Math.sqrt(result_avg_sq_mat[i][j])+eps));
        }
        result.push(r);
    }

    return [result_avg_sq_mat, result];
}

function scaleMat(scale, mat) {
    var result = [];
    for (var i=0; i<mat.length; i++) {
        var r = [];
        for (var j=0; j<mat[i].length; j++) {
            r.push(scale*mat[i][j]);
        }
        result.push(r);
    }
    return result;
}

function transpose(mat) {
    var result = [];
    for (var j=0; j<mat[0].length; j++) {
        var row = [];
        for (var i=0; i<mat.length; i++) {
            row.push(mat[i][j]);
        }
        result.push(row);
    }
    return result;
}

function copy_mat(mat) {
    var result = [];
    for (var i=0; i<mat.length; i++) {
        var row = [];
        for (var j=0; j<mat.length; j++) {
            row.push(mat[i][j]);
        }
        result.push(row);
    }
    return result;
}

// data
var trainingData = [];
var validationData = [];
var iris2index = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2};

// hyper-params
var eta = 0.1;
// rmsprop params
var eps = 0.00000001;
var gamma = 0.9;
var net1_avg_grad_sq_w1, net1_avg_grad_sq_b1;
var net2_avg_grad_sq_w1, net2_avg_grad_sq_b1, net2_avg_grad_sq_w2, net2_avg_grad_sq_b2;

// net-1 params
var net1mat1;
var net1bias1;

// net-2 params
var net2mat1;
var net2bias1;
var net2mat2;
var net2bias2;

// net input/output values - needed for backprop.
var x, z1, y1, z2, y2, yhat, ystar;

/* Load data for training and testing */
$.ajax({
    url: "iris-data.txt",
    success: function(data) { 
        var lines = data.split(/\r?\n/);
        for (var i=0; i<lines.length; i++) {
            var line = lines[i].split(",");
            if (line.length === 5) {
                var sample = line.slice(0, 4);
                sample.push(iris2index[line[4]]);
                if (i%3==0) {
                    validationData.push(sample);
                } else {
                    trainingData.push(sample);
                }
            }
        }
    }
});



