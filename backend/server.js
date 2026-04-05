const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");

const app = express();
app.use(express.json());

const server = http.createServer(app);

const io = new Server(server, {
    cors: {
        origin: "*"
    }
});

io.on("connection", (socket) => {

    console.log("Client connected");

    socket.on("stroke_data", async (data) => {

        console.log("Received strokes");

        const strokes = data.drawing;

        // ✅ Log correctly
        console.log(JSON.stringify(strokes, null, 2));

        try {
            const response = await axios.post(
                "http://localhost:5000/predict",
                { strokes: strokes }
            );

            const prediction = response.data.prediction;

            socket.emit("prediction", {
                label: prediction
            });

        } catch (error) {

            console.error("Prediction error:", error.message);

            socket.emit("prediction", {
                label: "Error"
            });

        }

    });

    socket.on("disconnect", () => {
        console.log("Client disconnected");
    });

});

server.listen(3000, () => {
    console.log("Server running on port 3000");
});