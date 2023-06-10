function image_reflector(input) {
    console.log(input)
    if (input.files.length === 0) return 0;
    let image = new Image();
    image.src = URL.createObjectURL(input.files[0]);
    document.getElementById("canvas_div").innerHTML = "";
    const canvas_element = document.createElement('canvas');
    canvas_element.id = "canvas"
    canvas_element.classList.add('uk-align-center')
    document.getElementById("canvas_div").appendChild(canvas_element);
    send_predict(input.files[0]);
    document.getElementById("predict_content").innerText = "\n\n\n\n\n\n";

}


const send_predict = (image) => {
    const file_descriptor = new FormData();
    file_descriptor.append('file', image);
    fetch('/hello_image_recog', {
        method: 'POST', body: file_descriptor
    }).then(response => response.json()).then(data => {
        console.log(data);
        fab_js(data, URL.createObjectURL(image));
    }).catch((error) => {
        console.error(error)
    })
};

function predict_view(content) {
    document.getElementById("predict_content").innerText = "\n\n\n\n\n\n";
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
    //const pos = document.getElementById("wid");
    //const targetRect = pos.getBoundingClientRect();
    //const targetTop = targetRect.top + window.scrollY;
    //window.scrollTo({
    //    top: targetTop,
    //    behavior: "auto"
    //})
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
        canvas.renderAll(() => {
            console.log("aaaa");
        });
        let canvas_element = document.getElementById("canvas");
        const centering_pixel = `${(document.getElementById('wid').clientWidth - document.getElementById("canvas").clientWidth) / 2}px`;
        canvas_element.style.marginLeft = centering_pixel;
        canvas_element = canvas_element.nextElementSibling;
        canvas_element.style.marginLeft = centering_pixel;
        console.log("resized!");
    }


    const canvas = new fabric.Canvas("canvas", {selection: false});
    window.addEventListener('resize', () => {
        clearTimeout(resize_timer);
        resize_timer = setTimeout(() => {
            resizeCanvas(canvas, Math.min(image.naturalWidth, 1000));
        }, 100);
    });
    canvas.setBackgroundImage(img, (e) => {
        canvas.setDimensions({
            width: e.width, height: e.height
        });
        resizeCanvas(canvas, Math.min(image.naturalWidth, 1000));
        canvas.renderAll()
    });
    if (data.count === 0) {
        console.log("no faces");
    } else {
        let item_order = 0;
        data.faces.forEach((item) => {
            //console.log(item);
            const conf = {
                id: item_order,
                originX: "center",
                originY: "center",
                left: (item.bbox[0] + item.bbox[2]) / 2,
                top: (item.bbox[1] + item.bbox[3]) / 2,
                width: item.bbox[2] - item.bbox[0],
                height: item.bbox[3] - item.bbox[1],
                fill: 'rgba(0,0,0,0)',
                strokeWidth: 3,
                stroke: 'rgba(255,0,0,1)',
                angle: 360 * item.rotate / (Math.PI * 2),
                hasControls: false,
                lockMovementY: true,
                lockMovementX: true,
                borderScaleFactor: 6
            }
            item_order++;
            //console.log(conf);
            const faceRect = new fabric.Rect(conf);
            canvas.add(faceRect);

            canvas.on({
                'selection:updated': (x) => {
                    predict_view(data.faces[x.selected[0].id].pred);
                },
                'selection:created': (x) => {
                    predict_view(data.faces[x.selected[0].id].pred);
                }
            })
        })
    }
    canvas.renderAll();
}


const wrong_report = () => {
    const file_descriptor = new FormData();
    const image = document.getElementById("image_selector").files[0];
    file_descriptor.append('file', image);
    fetch('/wrong_report', {
        method: 'POST', body: file_descriptor
    }).then(response => response.json()).then(data => {
        console.log(data);
    }).catch((error) => {
        console.error(error)
    })
};