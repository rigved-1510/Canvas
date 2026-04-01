const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const socket = io("http://localhost:3000");

let drawing = false;

let strokes = [];
let currentStroke = [];

let lastSentTime = 0;

/* Start drawing */

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
/* Continue drawing */

canvas.addEventListener("mousemove", (e)=>{

    if(!drawing) return;

    const rect = canvas.getBoundingClientRect();

    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    ctx.lineTo(x,y);
    ctx.stroke();

    currentStroke.push([x,y]);

    throttleSend();

});
/* End stroke */

canvas.addEventListener("mouseup", ()=>{

    drawing = false;

    if(currentStroke.length > 0)
        strokes.push(currentStroke);

});
/* Clear canvas */

function clearCanvas(){

    ctx.clearRect(0,0,canvas.width,canvas.height);

    strokes = [];
    currentStroke = [];

}
/* Convert strokes to QuickDraw format */

function convertToQuickDrawFormat(){

    let formatted = [];

    let allStrokes = [...strokes];

    // include current stroke while drawing
    if(currentStroke.length > 0){
        allStrokes.push(currentStroke);
    }

    allStrokes.forEach(stroke => {

        let xs = [];
        let ys = [];

        stroke.forEach(point => {

            xs.push(point[0]);
            ys.push(point[1]);

        });

        formatted.push([xs, ys]);

    });

    return formatted;
}
/* Throttle sending strokes */

function throttleSend(){

    const now = Date.now();

    if(now - lastSentTime < 250)
        return;

    lastSentTime = now;

    sendStrokes();

}
/* Send strokes to backend */

function sendStrokes(){

    const drawingData = convertToQuickDrawFormat();

    socket.emit("stroke_data", {
        drawing: drawingData
    });

}
/* Receive prediction */

socket.on("prediction", (data)=>{

    document.getElementById("prediction").innerText =
        "Prediction: " + data.label;

});