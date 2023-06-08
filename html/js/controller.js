let _input;
let recognition_data;

function image_reflector(input) {
    console.log(input)
    if (input.files.length === 0) return 0;
    let image = new Image();
    image.src = URL.createObjectURL(input.files[0]);
    _input = input;
    document.getElementById("image_canvas").innerHTML = "";
    document.getElementById("canvas_second").innerHTML = "";
    const canvas_element = document.createElement('canvas');
    canvas_element.id = "canvas__second"
    document.getElementById("canvas_second").appendChild(canvas_element);
    send_predict(input.files[0]);
    document.getElementById("predict_content").innerText = "";

}


const send_predict = (image) => {
    const file_descriptor = new FormData();
    file_descriptor.append('file', image);
    fetch('/hello_image_recog', {
        method: 'POST', body: file_descriptor
    }).then(response => response.json()).then(data => {
        recognition_data = data;
        console.log(data);
        fab_js(data, URL.createObjectURL(image));
    }).catch((error) => {
        console.error(error)
    })
};

function predict_view(content) {
    document.getElementById("predict_content").innerText = "";
    if (Object.keys(content.stat)[0] === "success") {
        for (const [k, v] of Object.entries(content)) {
            if (k === "stat") continue;
            let person = Object.keys(v)[0];
            let proba = Object.values(v)[0];
            document.getElementById("predict_content").innerText += `${parseInt(k) + 1}位:${person}  ${(proba * 100).toFixed(2)}%\n`;
        }
    } else if (Object.keys(content.stat)[0] === "invalid") {
        document.getElementById("predict_content").innerText = content.stat.invalid;
    }
}

let resize_timer;

const fab_js = (data, img) => {
    let image = new Image();
    image.src = img;
    let disp_width, disp_height;
    if (document.getElementById('wid').clientWidth < image.naturalWidth) {
        disp_width = document.getElementById('wid').clientWidth;
        disp_height = disp_width * (image.naturalHeight / image.naturalWidth);
    } else {
        disp_width = image.naturalWidth;
        disp_height = image.naturalHeight;
    }
    const resizeCanvas = (canvas, max_width) => {
        const real_width = Math.min(document.getElementById('wid').clientWidth, max_width);
        const ratio = canvas.getWidth() / canvas.getHeight();
        const scale = real_width / canvas.getWidth();
        const zoom = canvas.getZoom() * scale;
        canvas.setDimensions({
            width: real_width, height: real_width / ratio
        });
        canvas.setViewportTransform([zoom, 0, 0, zoom, 0, 0]);
        canvas.renderAll();
        console.log("resized!");
    }


    console.log(disp_width, disp_height);
    const canvas = new fabric.Canvas("canvas__second", {selection: false});
    window.addEventListener('resize', () => {
        clearTimeout(resize_timer);
        resize_timer = setTimeout(() => {
            //resizeCanvas(canvas, image.naturalWidth);
        }, 100);
    });
    canvas.setBackgroundImage(img, (e) => {
        canvas.setDimensions({
            width: e.width,
            height: e.height
        });
        resizeCanvas(canvas, image.naturalWidth);
        canvas.renderAll()
    });
    if (data.count === 0) {
        console.log("no faces");
    } else {
        data.faces.forEach((item) => {
            console.log(item);
            const conf = {
                originX: "center",
                originY: "center",
                left: (item.bbox[0] + item.bbox[2]) / 2,
                top: (item.bbox[1] + item.bbox[3]) / 2,
                width: item.bbox[2] - item.bbox[0],
                height: item.bbox[3] - item.bbox[1],
                fill: 'rgba(0,0,0,0)',
                strokeWidth: 3,
                stroke: 'rgba(255,0,0,1)',
                angle: 360 * item.rotate / (Math.PI * 2)
            }
            console.log(conf);
            const faceRect = new fabric.Rect(conf);
            canvas.add(faceRect);
            console.log("here1");
        })
    }
    canvas.renderAll();
}