let _input;
let recognition_data;

function image_reflector(input) {
    console.log(input)
    if (input.files.length === 0) return 0;
    let image = new Image();
    image.src = URL.createObjectURL(input.files[0]);
    _input = input;
    document.getElementById("image_canvas").innerHTML = "";
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
        new p5(plane, "image_canvas");
    }).catch((error) => {
        console.error(error)
    })
};

let clicked = 0;
const plane = (p) => {
    let img;
    p.preload = () => {
        img = p.loadImage(URL.createObjectURL(_input.files[0]));
    }
    let face_group = [];
    p.setup = () => {
        p.frameRate(4);
        let cnv = p.createCanvas(img.width, img.height);
        let real_width = document.getElementById("wid").clientWidth;
        real_width = Math.min(real_width, 1000);
        let scale = real_width / img.width;
        cnv.style('height', img.height * scale + 'px');
        cnv.style('width', real_width + 'px');
        cnv.style('margin', 'auto');
        console.log(scale);
        console.log(img.width, img.height);
        p.image(img, 0, 0);
        if (recognition_data !== undefined) {
            if (recognition_data.count === 0) return;
            p.noFill();
            p.strokeWeight(3);
            recognition_data.faces.forEach((elm) => {
                //console.log(elm.bbox);
                console.log(Object.keys(elm.pred.stat)[0])

                b = elm.bbox;
                let angle = elm.rotate;
                //console.log(angle);
                p.stroke(p.color("red"));
                //p.fill(p.color('transparent'));
                let sp = new p.Sprite(b[0] + (b[2] - b[0]) / 2, b[1] + (b[3] - b[1]) / 2, b[2] - b[0], b[3] - b[1], 'static');
                sp.rotation = 360 * angle / (2 * Math.PI);
                //sp.color = 'transparent';

                sp.color.setAlpha(0);
                if (Object.keys(elm.pred.stat)[0] === "success") {
                    sp.textSize = 20;
                    //sp.text = Object.keys(elm.pred[0])[0]
                }
                face_group.push(sp);

            })

        }
        //p.noLoop();
    };
    p.draw = () => {
        clicked--;
        if (-10 < clicked && clicked < 0) {
            face_group.forEach(sp => {
                sp.stroke = 'red';

            })
        }

    };
    p.mouseClicked = () => {
        if ((0 <= p.mouseX && p.mouseX <= p.width) && (0 <= p.mouseY && p.mouseY <= p.height) &&
            face_group.length === recognition_data.faces.length) {
            const euclid = face_group.map(sp => {
                return Math.sqrt(Math.pow(sp.position.x - p.mouseX, 2) + Math.pow(sp.position.y - p.mouseY, 2))
            })
            const dist = face_group.map(sp => {
                return Math.sqrt(Math.pow(sp.width / 2, 2) + Math.pow(sp.height / 2, 2))
            })
            //console.log(euclid, dist);
            for (let i = 0; i < face_group.length; i++) {
                console.log(face_group.length, recognition_data.faces.length);
                if (euclid[i] < dist[i]) {
                    console.log(i);

                    console.log(recognition_data.faces[i].pred);
                    predict_view(recognition_data.faces[i].pred);
                    face_group[i].stroke = p.color('blue');
                    clicked = 4;
                }
            }
        }
    }
};

let resize_timer;

function canvas_resize() {
    document.getElementById("image_canvas").innerHTML = "";
    new p5(plane, "image_canvas");
}

window.addEventListener('resize', () => {
    clearTimeout(resize_timer);
    resize_timer = setTimeout(canvas_resize, 600);
    console.log("resized");
});

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