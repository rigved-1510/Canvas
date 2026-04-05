const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const socket = io("http://localhost:3000");

let drawing = false;

let strokes = [];
let currentStroke = [];

/* ---------------- UTIL FUNCTIONS ---------------- */

// Downsample (reduce points)
function downsampleStroke(stroke, step = 3) {
    return stroke.filter((_, i) => i % step === 0);
}

// Smooth stroke
function smooth(arr) {
    return arr.map((v, i, a) => {
        if (i === 0 || i === a.length - 1) return v;
        return (a[i-1] + v + a[i+1]) / 3;
    });
}

/* ---------------- DRAWING ---------------- */

canvas.addEventListener("mousedown", (e)=>{

    drawing = true;
    currentStroke = [];

    const rect = canvas.getBoundingClientRect();

    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    ctx.beginPath();
    ctx.moveTo(x,y);

    currentStroke.push([x,y]);

});

canvas.addEventListener("mousemove", (e)=>{

    if(!drawing) return;

    const rect = canvas.getBoundingClientRect();

    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    ctx.lineTo(x,y);
    ctx.stroke();

    currentStroke.push([x,y]);

});

canvas.addEventListener("mouseup", ()=>{

    drawing = false;

    if(currentStroke.length > 0)
        strokes.push(currentStroke);

    // 🔥 SEND ONLY AFTER DRAWING COMPLETE
    sendStrokes();

});

/* ---------------- CLEAR ---------------- */

function clearCanvas(){

    ctx.clearRect(0,0,canvas.width,canvas.height);

    strokes = [];
    currentStroke = [];

}

/* ---------------- CONVERT + FIX INPUT ---------------- */

function convertToQuickDrawFormat(){

    let formatted = [];

    let allStrokes = [...strokes];

    if(currentStroke.length > 0){
        allStrokes.push(currentStroke);
    }

    const rect = canvas.getBoundingClientRect();

    allStrokes.forEach(stroke => {

        // limit length
        if (stroke.length > 200) {
            stroke = stroke.slice(0, 200);
        }

        // downsample
        stroke = downsampleStroke(stroke, 5);

        let xs = [];
        let ys = [];

        stroke.forEach(point => {

            let scaledX = (point[0] / rect.width) * 255;
            let scaledY = (point[1] / rect.height) * 255;

            xs.push(scaledX);
            ys.push(scaledY);

        });

        // smooth
        xs = smooth(xs);
        ys = smooth(ys);

        // round
        xs = xs.map(v => Math.round(v));
        ys = ys.map(v => Math.round(v));

        formatted.push([xs, ys]);

    });

    return formatted;
}

/* ---------------- SEND ---------------- */

function sendStrokes(){

    const drawingData = convertToQuickDrawFormat();

    socket.emit("stroke_data", {
        drawing: drawingData
    });

}

/* ---------------- RECEIVE ---------------- */

socket.on("prediction", (data)=>{

    document.getElementById("prediction").innerText =
        "Prediction: " + data.label;

});