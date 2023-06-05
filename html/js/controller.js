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

}


const send_predict = (image) => {
    const file_descriptor = new FormData();
    file_descriptor.append('file', image);
    fetch('/hello_image_recog', {
        method: 'POST', body: file_descriptor
    }).then(response => response.json()).then(data => {
        recognition_data = data;
        new p5(plane, "image_canvas");
        console.log(data);
    }).catch((error) => {
        console.error(error)
    })
};

const plane = (p) => {
    let img;
    p.preload = () => {
        img = p.loadImage(URL.createObjectURL(_input.files[0]));
    }
    let face_group;
    p.setup = () => {
        p.frameRate(4);
        p.noLoop();
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
            face_group = face_rect(p, recognition_data);
            console.log(face_group);
        }
        p.loop();
    };
    p.draw = () => {
        if (face_group !== undefined) {
            face_group.draw();
        }
        //p.circle(p.mouseX, p.mouseY, 20);
    };
};

const face_rect = (p, data) => {
    if (data.count === 0) return;
    p.noFill();
    p.strokeWeight(3);
    let face_group = new p.Group();
    data.faces.forEach((elm) => {
        console.log(elm.bbox);

        b = elm.bbox;
        let angle = elm.rotate;
        console.log(angle);
        p.stroke(p.color("red"));
        //p.fill(p.color('transparent'));
        let sp = new p.Sprite(b[0] + (b[2] - b[0]) / 2, b[1] + (b[3] - b[1]) / 2, b[2] - b[0], b[3] - b[1], 'static');
        sp.rotation = 360 * angle / (2 * Math.PI);
        //sp.color = 'transparent';
        sp.color.setAlpha(0);
        sp.onMouseOver = function () {
            console.log("pressed!!");
        }
        face_group.add(sp);

        //p.push();
        //p.translate(b[0] + (b[2] - b[0]) / 2, b[1] + (b[3] - b[1]) / 2);
        //p.rotate(360 * angle / (2 * Math.PI));
        //p.rect(-(b[2] - b[0]) / 2, -(b[3] - b[1]) / 2, b[2] - b[0], b[3] - b[1]);
        //p.pop();
    })
    return face_group;
}

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
    document.getElementById("predict_content").innerText = content;
}