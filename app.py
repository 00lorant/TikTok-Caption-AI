import os
import tempfile
import base64
import json
import logging

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse 
from openai import OpenAI

app = FastAPI()




@app.get("/")
async def home():
    return FileResponse("index.html")
    
@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="The AI API key has not been configured."
        )
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file."
        )

    temp_path = None
    cap = None

    try:
        suffix = os.path.splitext(video.filename or "video.mp4")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_path = temp.name
            while chunk := await video.read(1024 * 1024):
                temp.write(chunk)

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not read this video."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0 or fps <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not read the video length."
            )

        duration = total_frames / fps

        # Sample up to 6 frames from across the video.
        frame_count = min(6, max(1, int(duration / 5) + 1))
        timestamps = [
            duration * i / max(1, frame_count - 1)
            for i in range(frame_count)
        ]

        images = []

        for timestamp in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = cap.read()

            if not success:
                continue

            height, width = frame.shape[:2]
            scale = min(640 / width, 640 / height, 1)
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale))
            )

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )

            if success:
                image_data = base64.b64encode(
                    encoded.tobytes()
                ).decode("utf-8")

                images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })

        if not images:
            raise HTTPException(
                status_code=400,
                detail="Could not extract images from this video."
            )

        prompt = [
            {
                "type": "text",
                "text": (
                    "Analyze these frames from a video in their original order. "
                    "Understand what the video is about. Create one engaging, "
                    "natural TikTok caption that accurately describes the video. "
                    "Also recommend 5 relevant hashtags. Do not invent events "
                    "that are not visible. Return ONLY valid JSON in this format: "
                    '{"caption":"Your caption here","hashtags":["#tag1","#tag2","#tag3","#tag4","#tag5"]}'
                )
            }
        ]

        prompt.extend(images)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )

        result = json.loads(response.choices[0].message.content)

        return JSONResponse({
            "caption": result.get("caption", ""),
            "hashtags": result.get("hashtags", [])
        })

    except HTTPException:
        raise

    except Exception as error:
        logging.exception("Video analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Video analysis failed. Please try again."
        )

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
