import numpy as np
import cv2
from model import get_model, FRAMES_PER_VIDEO
from preprocess import video_to_npy,normalize

model = get_model()

threshold = 0.5

def embed_video(video_in, video_out):
    proba = classify(video_in)

    cap = cv2.VideoCapture(video_in) 
    (w,h) = (int(cap.get(3)),int(cap.get(4)))

    # h264 for web browsers compatibility 
    # https://github.com/cisco/openh264/releases/tag/v1.8.0
    # DOWNLOAD AND ADD TO PATH 
    out = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc(*'h264'), cap.get(5), (w,h))

    is_border = False
    for i in range(int(cap.get(7))):
        if i % 10 == 0: 
            is_border = not is_border
        current_prob = proba[i // FRAMES_PER_VIDEO][0]

        ret, frame = cap.read() 

        if current_prob[0] > threshold:
            if is_border:               
                frame = _draw_border(frame)

        frame = _draw_text(frame,f'Fight: {current_prob[0]:1.2f}')

        out.write(frame)

    cap.release()
    out.release()
    print(f'Successfuly written with codec h264!')

def _draw_border(image):
    h,w,_ = image.shape
    start_point = (0, 0)
    end_point = (w,h)
    thickness = h // 15
    image = cv2.rectangle(image, start_point, end_point, (0,0,255), thickness) 
    return image

def _draw_text(image, text):
    h,w,_ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX 
    scale = 1
    thickness = 1
    color = (0,255,255)
    (text_w,text_h), base = cv2.getTextSize(text, font, scale, thickness)
    while text_w > w:
        scale -= 0.1
        (text_w,text_h), base = cv2.getTextSize(text, font, scale, thickness)


    start_point = (w-text_w-(h // 15), h // 15)
    end_point = (start_point[0] + text_w, start_point[1] + text_h)

    image = cv2.rectangle(image, start_point, end_point, (0,0,0), -1) 
    cv2.putText(image, text, 
                    (start_point[0], start_point[1] + text_h),  
                    font, scale,  
                    color, thickness)
    return image

def classify(video_path):
    video = video_to_npy(video_path)
    chunks = _get_video_chunks(video)
    return [model(chunk.reshape(1,64,224,224,5),training=False) for chunk in chunks]
    
def _get_video_chunks(video):
    chunks_count = (video.shape[0] // FRAMES_PER_VIDEO) + 1
    to_pad = (chunks_count * FRAMES_PER_VIDEO) - video.shape[0]
    video = np.concatenate((video,video[-to_pad:,...]))
    chunks = []
    for i in range(chunks_count):
        data = video[i*FRAMES_PER_VIDEO:(i + 1)*FRAMES_PER_VIDEO,...]
        # normalize rgb images and optical flows, respectively
        data[...,:3] = normalize(data[...,:3])
        data[...,3:] = normalize(data[...,3:])
        chunks.append(data)
    return np.array(chunks)