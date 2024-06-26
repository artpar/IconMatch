import argparse
import random as rng
import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from icondetection.box import grayscale_blur, canny_detection, group_rects, candidate_rectangle
from icondetection.rectangle import Rectangle

app = Flask(__name__)

def closest_rectangle_handler(event: int, x: int, y: int, flags, params) -> None:
    global src, src2, candidate_rect, grouped_rects, excluded_rects

    if event == cv.EVENT_LBUTTONDOWN:
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

        src2 = src.copy()
        candidate_rect = candidate_rectangle([Rectangle.rect_cv_to_cartesian(t) for t in grouped_rects], (y, x))
        excluded_rects = filter(lambda rect: rect is not candidate_rect, grouped_rects)

        cv.rectangle(
            src2,
            (candidate_rect.bottom, candidate_rect.left),
            (candidate_rect.top, candidate_rect.right),
            color,
            2,
        )

def null_handler(event: int, x: int, y: int, flags, params) -> None:
    pass

def candidate_rectangle_demo() -> None:
    pass

def render_rectangles(rectangles, input_image, display_text, callback=null_handler, desired_color: tuple = None) -> None:
    for index in range(len(rectangles)):
        color = desired_color if desired_color is not None else (
            rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)
        )
        cv.rectangle(
            input_image,
            (int(rectangles[index][0]), int(rectangles[index][1])),
            (
                int(rectangles[index][0] + rectangles[index][2]),
                int(rectangles[index][1] + rectangles[index][3]),
            ),
            color,
            2,
        )
    return input_image

def threshold_callback(multiplier: int, contour_accuracy: int, min_threshold: int, src):
    gray_scale_image = grayscale_blur(src)
    _, bound_rect = canny_detection(gray_scale_image, multiplier=multiplier, contour_accuracy=contour_accuracy, min_threshold=min_threshold)
    grouped_rects = group_rects(bound_rect, 0, src.shape[1])

    src_copy = src.copy()
    result_img = render_rectangles(grouped_rects, src_copy, "Grouped Rectangles", desired_color=(36, 9, 14))
    return result_img

def threshold_callback_rects(multiplier: int, contour_accuracy: int, min_threshold: int, src):
    gray_scale_image = grayscale_blur(src)
    _, bound_rect = canny_detection(gray_scale_image, multiplier=multiplier, contour_accuracy=contour_accuracy, min_threshold=min_threshold)
    grouped_rects = group_rects(bound_rect, 0, src.shape[1])
    return grouped_rects


@app.route('/process_image_json', methods=['POST'])
def process_image_json():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file provided'}), 400


    npimg = np.frombuffer(file.read(), np.uint8)
    src = cv.imdecode(npimg, cv.IMREAD_COLOR)

    if src is None:
        return jsonify({'error': 'Could not open or find the image'}), 400

    multiplier = int(request.values['multiplier'])
    contour_accuracy = int(request.values['contour_accuracy'])
    min_threshold = int(request.values['min_threshold'])
    grouped_rects = threshold_callback(multiplier, contour_accuracy, min_threshold, src)

    rects_json = []
    for rect in grouped_rects:
        if rect[1][0] - rect[0][0] == 0:
            continue
        if rect[1][1] - rect[0][1] == 0:
            continue
        rects_json.append({
            'x': int(rect[0][0]),
            'y': int(rect[0][1]),
            'width': int(rect[1][0] - rect[0][0]),
            'height': int(rect[1][1] - rect[0][1])
        })

    return jsonify({'rectangles': rects_json})

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file provided'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    src = cv.imdecode(npimg, cv.IMREAD_COLOR)

    if src is None:
        return jsonify({'error': 'Could not open or find the image'}), 400

    multiplier = int(request.values['multiplier'])
    contour_accuracy = int(request.values['contour_accuracy'])
    min_threshold = int(request.values['min_threshold'])
    result_img = threshold_callback(multiplier, contour_accuracy, min_threshold, src)

    _, buffer = cv.imencode('.jpg', result_img)
    img_bytes = BytesIO(buffer)

    return send_file(img_bytes, mimetype='image/jpeg')

if __name__ == "__main__":
    rng.seed(12345)
    app.run(host='0.0.0.0', port=5001)